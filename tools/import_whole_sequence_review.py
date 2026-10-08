"""Import consolidated order approval without inventing timing or changing old references."""
from pathlib import Path
import argparse
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,sha256_file,verify_file
from any2xsampa.training.whole_sequence_review import validate_answers


def ingest(packet_path:Path,answers_path:Path,output:Path)->None:
    packet=read_json(packet_path);answers=read_json(answers_path)
    result=validate_answers(packet,answers,packet_sha256=sha256_file(packet_path))
    for item in packet['items']:
        for source in item['source_bindings'].values():verify_file(Path(source['path']),source['sha256'])
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(answers_path,output/'raw-answers.json')
    write_json_new(output/'REVIEW_INTAKE.json',{**result,'packet_path':str(packet_path.resolve()),
        'packet_sha256':sha256_file(packet_path),'answers_sha256':sha256_file(answers_path),
        'old_reference_files_modified':False,'whole_training_dataset_created':False})
    print('accepted',len(result['accepted']),'pending',len(result['pending']),'conflicts',len(result['conflicts']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--answers',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();ingest(args.packet.resolve(),args.answers.resolve(),args.output.resolve())
