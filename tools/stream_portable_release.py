"""Write a portable ZIP straight to bounded volumes, without a full ZIP copy."""
from __future__ import annotations

import argparse
import bisect
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import stat
import time
import zipfile

GIB = 1024**3


class VolumeWriter(io.RawIOBase):
    def __init__(self, output: Path, name: str, part_bytes: int, maximum: int,
                 reserve: int) -> None:
        super().__init__()
        self.output, self.name = output, name
        self.part_bytes, self.maximum, self.reserve = part_bytes, maximum, reserve
        self.parts: list[dict] = []
        self.total = 0
        self.digest = hashlib.sha256()
        self.part_digest = hashlib.sha256()
        self.current = None
        self.size = 0
        self.next_space_check = 0

    def writable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.total

    def _finish_part(self) -> None:
        if self.current is not None:
            self.current.close()
            self.parts.append(dict(name=Path(self.current.name).name, bytes=self.size,
                                   sha256=self.part_digest.hexdigest()))
            self.current = None

    def write(self, data: bytes) -> int:
        if self.closed:
            raise ValueError("Volume writer is closed")
        if self.total + len(data) > self.maximum:
            raise OSError("Archive exceeds the frozen disk budget")
        view = memoryview(data)
        while view:
            if self.total >= self.next_space_check:
                if shutil.disk_usage(self.output).free < self.reserve + 64 * 1024**2:
                    raise OSError("Free disk space reached the reserved floor")
                self.next_space_check = self.total + 64 * 1024**2
            if self.current is None:
                name = self.name + f".{len(self.parts)+1:03d}"
                self.current = (self.output / name).open("xb")
                self.part_digest, self.size = hashlib.sha256(), 0
            block = view[:self.part_bytes - self.size]
            count = self.current.write(block)
            if count != len(block):
                raise OSError("Short volume write")
            self.digest.update(block)
            self.part_digest.update(block)
            self.total += count
            self.size += count
            view = view[count:]
            if self.size == self.part_bytes:
                self._finish_part()
        return len(data)

    def close(self) -> None:
        self._finish_part()
        super().close()


class VolumeReader(io.RawIOBase):
    """Seek across volume files for CRC verification, with one open handle."""
    def __init__(self, output: Path, parts: list[dict]) -> None:
        super().__init__()
        self.output, self.parts = output, parts
        self.starts = [0]
        for part in parts:
            self.starts.append(self.starts[-1] + part["bytes"])
        self.position = 0
        self.handle = None
        self.index = -1

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.position, 2: self.starts[-1]}
        if whence not in base or offset + base[whence] < 0:
            raise ValueError("Invalid volume seek")
        self.position = base[whence] + offset
        return self.position

    def read(self, size: int = -1) -> bytes:
        remaining = max(0, self.starts[-1] - self.position)
        count = remaining if size < 0 else min(size, remaining)
        chunks = []
        while count:
            index = bisect.bisect_right(self.starts, self.position) - 1
            if self.index != index:
                if self.handle is not None:
                    self.handle.close()
                self.handle = (self.output / self.parts[index]["name"]).open("rb")
                self.index = index
            self.handle.seek(self.position - self.starts[index])
            expected = min(count, self.starts[index+1] - self.position)
            chunk = self.handle.read(expected)
            if len(chunk) != expected:
                raise OSError("Volume was truncated during verification")
            chunks.append(chunk)
            self.position += len(chunk)
            count -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
        super().close()


def package_files(source: Path) -> list[Path]:
    if source.is_symlink() or source.is_junction():
        raise ValueError("Application root must not be a link")
    files = []
    for path in sorted(source.rglob("*")):
        rel = path.relative_to(source)
        if set(rel.parts) & {".state", "__pycache__"} or path.suffix.lower() in {".pyc", ".pdb"}:
            continue
        attributes = path.lstat()
        if (stat.S_ISLNK(attributes.st_mode)
                or getattr(attributes, "st_file_attributes", 0) & 0x400):
            raise ValueError("Application payload contains a link: " + str(rel))
        if stat.S_ISREG(attributes.st_mode):
            files.append(path)
    if not files:
        raise ValueError("Application payload is empty")
    return files


def stream_release(source: Path, output: Path, release_tag: str, *,
                   part_bytes: int = GIB, max_archive_bytes: int = 6*GIB,
                   reserve_bytes: int = 30*GIB,
                   include_paths: list[str] | None = None) -> dict:
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", release_tag):
        raise ValueError("Expected a stable release tag")
    if not 0 < part_bytes < 2*GIB or max_archive_bytes <= 0 or reserve_bytes < 0:
        raise ValueError("Invalid volume or disk budget")
    source, output = source.absolute(), output.resolve()
    if output.is_relative_to(source.resolve()):
        raise ValueError("Volumes must be outside the application payload")
    name = source.name + ".zip"
    manifest = output / "release-manifest.json"
    if manifest.exists() or any(output.glob(name + ".*")):
        raise FileExistsError("Release volumes already exist; no automatic retry")
    files = package_files(source)
    if include_paths is not None:
        available = {p.relative_to(source).as_posix(): p for p in files}
        if (not include_paths or len({p.casefold() for p in include_paths}) != len(include_paths)
                or any(p not in available for p in include_paths)):
            raise ValueError("Explicit payload inventory is invalid or missing a file")
        # An installed app may already contain weights; package only the reviewed
        # program inventory so no second full staging tree is necessary.
        files = sorted(available[p] for p in include_paths)
    sizes = {p: p.stat().st_size for p in files}
    if shutil.disk_usage(output).free < reserve_bytes + max_archive_bytes:
        raise OSError("Insufficient space for archive budget and reserve")
    started, last = time.monotonic(), time.monotonic()
    writer = VolumeWriter(output, name, part_bytes, max_archive_bytes, reserve_bytes)
    try:
        with zipfile.ZipFile(writer, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as archive:
            for index, path in enumerate(files):
                archive.write(path, (Path(source.name) / path.relative_to(source)).as_posix())
                if path.stat().st_size != sizes[path]:
                    raise ValueError("Payload changed during compression")
                if time.monotonic() - last > 20:
                    print(index+1, len(files), round(writer.total/1024**2), "MiB", flush=True)
                    last = time.monotonic()
    finally:
        writer.close()
    full = hashlib.sha256()
    for part in writer.parts:
        digest = hashlib.sha256()
        with (output / part["name"]).open("rb") as handle:
            while block := handle.read(1024**2):
                full.update(block)
                digest.update(block)
        if digest.hexdigest() != part["sha256"]:
            raise ValueError("Written volume failed SHA256 verification")
    if full.hexdigest() != writer.digest.hexdigest():
        raise ValueError("Written archive failed SHA256 verification")
    with VolumeReader(output, writer.parts) as reader, zipfile.ZipFile(reader) as archive:
        entries = archive.infolist()
        expected = [(Path(source.name)/p.relative_to(source)).as_posix() for p in files]
        if [e.filename for e in entries] != expected or sum(e.file_size for e in entries) != sum(sizes.values()):
            raise ValueError("Written archive inventory mismatch")
        if archive.testzip() is not None:
            raise ValueError("Written archive failed CRC verification")
    value = dict(schema_version="portable_split_release_v2", root_directory=source.name,
                 archive_bytes=writer.total, archive_sha256=full.hexdigest(),
                 expanded_bytes=sum(sizes.values()), file_count=len(files), parts=writer.parts,
                 release_tag=release_tag,
                 download_base_url="https://github.com/Hoshino-Rubi/any2xsampa/releases/download/"+release_tag+"/")
    with manifest.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    print("STREAM_VERIFIED", len(files), round(time.monotonic()-started, 2), flush=True)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--release-tag", required=True)
    args = parser.parse_args()
    print(json.dumps(stream_release(args.source, args.output, args.release_tag)))


if __name__ == "__main__":
    main()
