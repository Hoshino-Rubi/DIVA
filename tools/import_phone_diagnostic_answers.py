from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from any2xsampa.training.phone_diagnostic_review import import_answers
if __name__=='__main__':
    p=argparse.ArgumentParser(description='Import diagnostic answers; no relabeling or training.');p.add_argument('--packet',type=Path,required=True);p.add_argument('--answers',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not a.output.resolve().is_relative_to(ROOT/'.state/runs'):raise ValueError('new intake must be under runs')
    r=import_answers(a.packet,a.answers,a.output);print('Imported diagnostic answers:',len(r['responses']))
