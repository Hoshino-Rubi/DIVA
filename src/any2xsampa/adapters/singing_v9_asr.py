"""Offline full Whisper decoder; attention-DTW timestamps on its native output."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import median_filter

def phrase_ranges(rows,duration):
    lo=None;end=0.
    for n in rows:
        a,b=n["start_sec"],n["end_sec"]
        if lo is not None and (a-end>.55 or b-lo>13.5):
            yield max(0.,lo-.13),min(duration,end+.09)
            lo=None
        if lo is None:lo=a
        end=b
    if lo is not None:yield max(0.,lo-.13),min(duration,end+.09)

def words_from_pieces(parts):
    text="".join(t for t,*_ in parts)
    offsets=np.cumsum([0]+[len(t) for t,*_ in parts])
    out=[]
    for m in re.finditer(r"[A-Za-z]+[1-6]?|[\u3400-\u9fff]+",text):
        a,b=m.span()
        cols=[k for k in range(len(parts)) if offsets[k]<b and offsets[k+1]>a]
        word=m[0]
        if re.fullmatch(r"[A-Za-z]+[1-6]?",word):
            syl=[word.lower()]
        else:
            from pycantonese import characters_to_jyutping
            syl=[v for _,jp in characters_to_jyutping(word) for v in (jp or "?").split()]
        # Separate multi-character Han words by their token spans, not reference text.
        for j,s in enumerate(syl):
            start=a+(b-a)*j/len(syl); end=a+(b-a)*(j+1)/len(syl)
            sel=[k for k in cols if offsets[k]<end and offsets[k+1]>start]
            if sel:out.append(dict(jp=s,start=parts[sel[0]][1],end=parts[sel[-1]][2],
                       confidence=float(np.exp(np.mean([np.log(max(parts[k][3],1e-9)) for k in sel])))))
    return text,out

class LocalWhisper:
    def __init__(self,bundle:Path):
        from transformers import WhisperProcessor,WhisperForConditionalGeneration
        local=str(bundle/"small")
        self.p=WhisperProcessor.from_pretrained(local,local_files_only=True)
        self.model=WhisperForConditionalGeneration.from_pretrained(local,local_files_only=True,weights_only=True,
                         dtype=torch.float16,attn_implementation="eager").cuda().eval()
    @torch.inference_mode()
    def phrase(self,wave,offset):
        from transformers.models.whisper.generation_whisper import _dynamic_time_warping
        features=self.p.feature_extractor(wave,sampling_rate=16000,return_tensors="pt",return_attention_mask=True)
        mel=features.input_features.cuda().half()
        model=self.model
        ids=model.generate(mel,num_beams=3,do_sample=False,max_new_tokens=min(130,round(len(wave)/16000*7)+15),
                         language="zh",task="transcribe",return_timestamps=False,attention_mask=features.attention_mask.cuda(),
                         return_dict_in_generate=True).sequences
        if int(ids[0,0])!=model.config.decoder_start_token_id:raise ValueError("Missing decoder prefix")
        valid=[k for k,t in enumerate(ids[0].tolist()) if k and t<model.config.eos_token_id]
        if not valid:return "",[]
        y=model(input_features=mel,decoder_input_ids=ids[:,:-1],output_attentions=True,use_cache=False)
        n=max(1,min(1500,int(np.ceil(len(wave)/320))))
        positions=[k-1 for k in valid]
        att=torch.stack([y.cross_attentions[l][0,h,positions,:n].float() for l,h in model.generation_config.alignment_heads])
        z=(att-att.mean(-2,keepdim=True))/att.std(-2,keepdim=True,unbiased=False).clamp_min(1e-6)
        path,ts=_dynamic_time_warping(-median_filter(z.mean(0).cpu().numpy(),size=(1,5)))
        probs=y.logits[0,positions].float().softmax(1).gather(1,ids[0,valid,None]).flatten().tolist()
        parts=[]
        for j,k in enumerate(valid):
            times=ts[path==j]
            if not len(times):continue
            token=self.p.tokenizer.decode([int(ids[0,k])],clean_up_tokenization_spaces=False)
            parts.append((token,offset+times[0]*.02,offset+min(len(wave)/16000,(times[-1]+1)*.02),probs[j]))
        return words_from_pieces(parts)

    def decode(self,audio,rows):
        out=[];torch.set_num_threads(4)
        for a,b in phrase_ranges(rows,len(audio)/16000):
            chunk=audio[round(a*16000):round(b*16000)]
            if len(chunk)<480 or not chunk.any():continue
            text,syllables=self.phrase(chunk,a)
            out.append(dict(a=a,b=b,text=text,syllables=syllables))
        return out
