"""Generate a public checksum list from the actual, explicitly named release assets."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

def sha(file: Path) -> str:
    with file.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def finalize(directory: Path, expected: list[str], tag: str) -> dict:
    if not re.fullmatch(r"(?:v[0-9]+\.[0-9]+\.[0-9]+|pre-[0-9]+\.[0-9]+\.[0-9]+[a-z]?)",tag):
        raise ValueError("Expected explicit release version tag")
    if len(expected)!=len(set(expected)) or "SHA256SUMS.txt" in expected:
        raise ValueError("Duplicate or self-referencing asset list")
    for name in expected:
        if Path(name).name!=name or "/" in name or "\\" in name:
            raise ValueError("Assets must be simple file names")
    actual={p.name for p in directory.iterdir() if p.is_file()}
    if actual!=set(expected):
        raise ValueError("Asset set mismatch; missing="+str(set(expected)-actual)+" extra="+str(actual-set(expected)))
    rows=[]
    for name in sorted(expected):
        file=directory/name
        if file.is_symlink() or not 0<file.stat().st_size<2*1024**3:
            raise ValueError("Invalid asset: "+name)
        rows.append(dict(name=name,bytes=file.stat().st_size,sha256=sha(file)))
    sums=directory/"SHA256SUMS.txt"
    with sums.open("x",encoding="utf-8",newline="\n") as stream:
        stream.write("".join(row["sha256"]+"  "+row["name"]+"\n" for row in rows))
    return dict(schema_version="release_assets_v1",tag=tag,assets=rows,
                checksums_sha256=sha(sums),checksum_file_excludes_itself=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",required=True,type=Path)
    p.add_argument("--files",required=True,type=Path,help="JSON array of explicit upload file names")
    p.add_argument("--tag",required=True)
    p.add_argument("--report",required=True,type=Path)
    p.add_argument("--require-installer-signature",action="store_true")
    p.add_argument("--signtool",type=Path)
    p.add_argument("--installer-name",default="Any2Xsampa-Setup.exe")
    a=p.parse_args()
    if a.report.exists():raise FileExistsError(a.report)
    if a.require_installer_signature:
        if Path(a.installer_name).name != a.installer_name or "/" in a.installer_name or "\\" in a.installer_name:
            raise ValueError("Installer must have a simple file name")
        if a.signtool is None or not a.signtool.is_file():raise ValueError("Trusted signing requires an explicit signtool executable")
        subprocess.run([str(a.signtool),"verify","/pa","/all",str(a.directory/a.installer_name)],check=True)
    result=finalize(a.directory,json.loads(a.files.read_text(encoding="utf-8")),a.tag)
    result["installer_signature_verified"]=a.require_installer_signature
    a.report.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
if __name__=="__main__":main()
