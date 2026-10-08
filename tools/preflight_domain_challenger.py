"""Check only the authorized proxy for immutable challenger metadata."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,urllib.request,urllib.error,importlib.util,hashlib

def preflight(output:Path)->None:
    output.mkdir(parents=True,exist_ok=False)
    proxy='http://127.0.0.1:10808'
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':proxy,'https':proxy}))
    models=[]
    # Query official metadata only. No weights, third-party code, installation or direct fallback.
    for model in ['espnet/powsm_ctc','espnet/powsm']:
        url='https://huggingface.co/api/models/'+model
        record={'model_id':model,'metadata_url':url,'checked_at_utc':datetime.now(timezone.utc).isoformat()}
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'any2xsampa-domain-preflight/1'})
            with opener.open(request,timeout=8) as response:raw=response.read(4*1024*1024+1)
            if len(raw)>4*1024*1024:raise ValueError('metadata exceeds bounded response')
            doc=json.loads(raw);revision=doc.get('sha')
            if not isinstance(revision,str) or len(revision)!=40:raise ValueError('no immutable revision in metadata')
            path=output/(model.replace('/','-')+'.metadata.json');path.write_bytes(raw)
            record.update(status='metadata_available',revision=revision,license=doc.get('cardData',{}).get('license'),
                          metadata_sha256=hashlib.sha256(raw).hexdigest())
        except (OSError,ValueError,urllib.error.URLError) as exc:
            record.update(status='blocked_proxy_or_metadata',error=str(exc))
        models.append(record)
        if record['status']=='blocked_proxy_or_metadata':break
    result={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'proxy':proxy,'direct_fallback':False,
        'models':models,'weights_downloaded':False,'dependencies_installed':False,
        'ordinary_environment_espnet2_available':importlib.util.find_spec('espnet2') is not None,
        'isolated_phonetic_environment_espnet2_available':False,
        'next_requirements':['immutable code/weights/tokenizer hashes','weight license confirmation','separate exact dependency lock','IPA inventory and normalization compatibility'],
        'ZIPA_status':'checkpoint_license_not_confirmed; no download','no_claim_of_real_challenger_inference':True}
    (output/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);preflight(p.parse_args().output.resolve())
