"""Install only the exact GAME archive authorized for note qualification; no network."""
from pathlib import Path
import argparse,hashlib,json,zipfile
from any2xsampa.music.game_notes import EXPECTED_ARCHIVE,REVISION,load_snapshot

def install(archive: Path,destination: Path) -> dict:
    digest=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()
    if digest!=EXPECTED_ARCHIVE:raise ValueError('GAME archive SHA256 mismatch')
    if destination.exists():raise FileExistsError('Refusing to replace a snapshot')
    with zipfile.ZipFile(archive) as z:
        members=[i for i in z.infolist() if not i.is_dir()]
        configs=[i for i in members if Path(i.filename).name=='config.json']
        if len(configs)!=1:raise ValueError('Ambiguous model config')
        prefix=Path(configs[0].filename).parent
        if sum(i.file_size for i in members)>1024**3:raise ValueError('Unexpected archive expansion')
        prepared=[]
        for i in members:
            p=Path(i.filename)
            if not p.is_relative_to(prefix):raise ValueError('Member outside model root')
            relative=p.relative_to(prefix)
            target=(destination/relative).resolve()
            if not target.is_relative_to(destination.resolve()) or (i.external_attr>>16)&0o170000==0o120000:
                raise ValueError('Unsafe snapshot path')
            prepared.append((i,target,relative))
        destination.mkdir(parents=True)
        files=[]
        for i,target,relative in prepared:
            target.parent.mkdir(parents=True,exist_ok=True)
            data=z.read(i)
            with target.open('xb') as f:f.write(data)
            files.append({'path':relative.as_posix(),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
    lock={'schema_version':'game_onnx_snapshot_v1','revision':REVISION,'archive_sha256':digest,'files':files}
    (destination/'snapshot-lock.json').write_text(json.dumps(lock,indent=2),encoding='utf8')
    _,config=load_snapshot(destination)
    return {'snapshot':str(destination),'config':config,'member_count':len(files),'archive_sha256':digest}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(install(a.archive,a.output),ensure_ascii=False))
if __name__=='__main__':main()
