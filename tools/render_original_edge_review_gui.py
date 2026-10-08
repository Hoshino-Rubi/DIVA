"""Render a packet as a self-contained, file://-usable human review page."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path
import wave

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).with_name('original_edge_review_gui')


def _inline_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False).replace('<', r'\u003c').replace('>', r'\u003e').replace('&', r'\u0026')


def render_packet(packet_dir: Path, packet_name: str = 'packet.json') -> Path:
    """Create index.html only; refuse overwrites and audio outside this new run."""
    packet_dir = packet_dir.resolve()
    if not packet_dir.is_relative_to((ROOT / '.state/runs').resolve()):
        raise ValueError('packet directory must be under .state/runs')
    output = packet_dir / 'index.html'
    if output.exists():
        raise FileExistsError(output)
    packet_path = (packet_dir / packet_name).resolve()
    if not packet_path.is_relative_to(packet_dir):
        raise ValueError('packet path escapes run directory')
    packet = json.loads(packet_path.read_text(encoding='utf-8'))
    if packet.get('schema_version') not in {'original_edge_review_packet_v1', 'replacement_edge_review_packet_v1'}:
        raise ValueError('unsupported packet schema')
    if not 1 <= len(packet.get('items', [])) <= 6:
        raise ValueError('packet must have 1–6 items')
    audio: dict[str, str] = {}
    for item in packet['items']:
        if item.get('cut_in_preview_sec') != 1.25 or item.get('duration_sec') != 2.5:
            raise ValueError('preview cut must be 1.25s within 2.5s')
        if not any(media.get('role') == 'original' for media in item['media']):
            raise ValueError('original media is required')
        for media in item['media']:
            name = media['audio_file']
            path = (packet_dir / name).resolve()
            if not path.is_relative_to(packet_dir / 'media'):
                raise ValueError('audio path must remain under run/media')
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != media['audio_sha256']:
                raise ValueError(f'audio sha256 mismatch: {name}')
            with wave.open(str(path), 'rb') as wav:
                if abs(wav.getnframes() / wav.getframerate() - 2.5) > 1 / wav.getframerate():
                    raise ValueError(f'audio must be a 2.5s preview: {name}')
            audio[name] = base64.b64encode(raw).decode('ascii')
    template = (ASSETS / 'index.html').read_text(encoding='utf-8')
    if packet['schema_version'] == 'replacement_edge_review_packet_v1':
        template = template.replace('沙龙 · 正式切点音素听审', '训练片段 · 正式切点音素听审')
    replacements = {
        'CSS': (ASSETS / 'style.css').read_text(encoding='utf-8'),
        'PACKET': _inline_json(packet), 'AUDIO': _inline_json(audio),
        'STATE': (ASSETS / 'state.js').read_text(encoding='utf-8'),
        'APP': (ASSETS / 'app.js').read_text(encoding='utf-8'),
    }
    template = re.sub(r'/[*] INLINE_([A-Z]+) [*]/', lambda match: replacements[match[1]], template)
    with output.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(template)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet-dir', type=Path, required=True)
    parser.add_argument('--packet-name', default='packet.json')
    args = parser.parse_args()
    print(render_packet(args.packet_dir, args.packet_name))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
