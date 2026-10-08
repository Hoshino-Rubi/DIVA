"""Render the five proxy-rule decisions as a standalone offline GUI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.final_test_mapping_review import verify_review_packet
from any2xsampa.training.limited_data_artifacts import sha256_file, write_json_new


def render(packet_path: Path) -> Path:
    packet=verify_review_packet(packet_path)
    assets=ROOT/'tools/final_mapping_review_gui'
    text=(assets/'index.html').read_text(encoding='utf-8-sig')
    values={'PACKET':json.dumps(packet,ensure_ascii=True).replace('<',r'\u003c'),
            'STATE':(assets/'state.js').read_text(encoding='utf-8-sig'),
            'APP':(assets/'app.js').read_text(encoding='utf-8-sig')}
    for key,value in values.items():text=text.replace('/* INLINE_'+key+' */',value)
    if '/* INLINE_' in text or text.count('<script')!=3 or text.count('</script>')!=3:
        raise ValueError('invalid generated HTML')
    output=packet_path.parent/'index.html'
    with output.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
    write_json_new(packet_path.parent/'GUI_BUILD.json',dict(
        packet_file_sha256=sha256_file(packet_path),packet_sha256=packet['packet_sha256'],
        index_sha256=sha256_file(output),
        assets={str(p.relative_to(ROOT)):sha256_file(p) for p in [assets/'index.html',assets/'app.js',assets/'state.js']},
        expected_answer='final-mapping-decisions-'+packet['packet_sha256'][:12]+'.json',
        browser_control_used=False,visual_browser_verification=False,test_predictions_opened=False))
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--packet',type=Path,required=True)
    print(render(parser.parse_args().packet.resolve()))
