"""Join independent audio pitch notes with IPA only after recognition."""
from __future__ import annotations
import math
from pathlib import Path
from xml.etree import ElementTree as ET
from .common import ROOT, read, sha, write
from .vsqx_export import _tokens_for_phone
from any2xsampa.vocaloid.draft import VocalNoteDraft, VocalTrackDraft, VocalTrackDraftTrack
from any2xsampa.vocaloid.phoneme_policy import validate_miku_v4c_note_phonemes
from any2xsampa.vocaloid.vsqx import build_vsqx_tree

POLICY = Path(__file__).resolve().parents[1] / "config/local_score_v1.json"

def note_number(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or int(value)!=value or not 0<=value<=127:
        raise ValueError("Pitch must be an integer MIDI note between 0 and 127")
    return int(value)

def positive(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
        raise ValueError(name+" must be finite and positive")
    return float(value)

def make_score(result: dict, pitch: dict, tempo: float) -> dict:
    tempo=positive(tempo,"tempo")
    policy=read(POLICY)
    rows=[]
    start,end=result["input"]["start_sec"],result["input"]["end_sec"]
    for index,raw in enumerate(pitch["note_candidates"]):
        begin=float(raw["start_sec"])
        duration=positive(raw["duration_sec"],"note duration")
        if not math.isfinite(begin) or begin<start-1e-5 or begin>=end:
            raise ValueError("Pitch note outside source interval")
        finish=min(end,begin+duration)
        if rows and begin < rows[-1]["end_sec"]-1e-5:
            raise ValueError("Pitch notes overlap; refusing a misleading monophonic score")
        rows.append({"index":index,"start_sec":begin,"end_sec":finish,"duration_sec":finish-begin,
                     "pitch":note_number(raw["midi"]),"pitch_source":"audio_pyin",
                     "phone_indices":[],"ipa":[],"v4c_phonemes":[],"approximations":[],
                     "unsupported_phones":[],"lyric":"", "placeholder":False})
    if not rows:
        raise ValueError("No voiced pitch notes found; no fixed-pitch substitute was created")
    unmatched=[]
    distance_limit=policy["alignment"]["maximum_phone_distance_sec"]
    for index,segment in enumerate(result["segments"]):
        midpoint=(float(segment["start_sec"])+float(segment["end_sec"]))/2
        candidates=[]
        for n,row in enumerate(rows):
            distance=max(row["start_sec"]-midpoint,midpoint-row["end_sec"],0)
            candidates.append((distance,abs((row["start_sec"]+row["end_sec"])/2-midpoint),n))
        distance,_,n=min(candidates)
        if distance>distance_limit:
            unmatched.append(index)
            continue
        row=rows[n]; phone=segment["normalized_ipa"]
        row["phone_indices"].append(index);row["ipa"].append(phone)
        tokens,approximation=_tokens_for_phone(phone,policy["v4c_phone_rules"])
        row["v4c_phonemes"].extend(tokens)
        if approximation:row["approximations"].append(phone)
        if not tokens:row["unsupported_phones"].append(phone)
    for row in rows:
        issues=validate_miku_v4c_note_phonemes(row["v4c_phonemes"])
        if not row["v4c_phonemes"] or row["unsupported_phones"] or issues:
            row["placeholder"]=True
            row["placeholder_reason"]="unsupported_or_missing_phone_evidence" if not issues else "known_invalid_v4c_sequence"
            row["proposed_v4c_phonemes"]=row["v4c_phonemes"]
            row["v4c_phonemes"]=policy["placeholder"]["phonemes"]
            row["lyric"]=policy["placeholder"]["lyric"]
        else:
            row["lyric"]="/"+" ".join(row["ipa"])+"/"
    return {"schema_version":"local_note_score_v1","source":"audio_pyin_draft","input":result["input"],
            "tempo":tempo,"tempo_source":"user_grid_tempo_not_estimated","ticks_per_quarter":policy["ticks_per_quarter"],
            "policy_sha256":sha(POLICY),"notes":rows,"phone_count":len(result["segments"]),
            "unassigned_phone_indices":unmatched,"unassigned_phone_count":len(unmatched),
            "placeholder_note_count":sum(r["placeholder"] for r in rows),
            "approximated_note_count":sum(bool(r["approximations"]) for r in rows),
            "pitch_transpose_semitones":0,"ctc_times_are_note_durations":False,
            "alignment_status":"automatic_draft_not_human_confirmed","model_forwards":0}

def _clean_template(template: Path, tempo: float, resolution: int) -> ET.ElementTree:
    tree=ET.parse(template);root=tree.getroot();ns=root.tag.split('}')[0]+'}'
    master=root.find(ns+'masterTrack')
    master.find(ns+'resolution').text=str(resolution)
    # One fixed grid; seconds are preserved by exact conversion, not beat quantization.
    for child in list(master):
        if child.tag in {ns+'tempo',ns+'timeSig'}:master.remove(child)
    time_sig=ET.SubElement(master,ns+'timeSig')
    for name,value in [('m',0),('nu',4),('de',4)]:ET.SubElement(time_sig,ns+name).text=str(value)
    event=ET.SubElement(master,ns+'tempo')
    ET.SubElement(event,ns+'t').text='0';ET.SubElement(event,ns+'v').text=str(round(tempo*100))
    # Audio-derived drafts use source seconds as the project origin. The
    # accepted template's pre-measure is useful for an imported project, but
    # would shift every generated note by one bar.
    master.find(ns+'preMeasure').text='0'
    tracks=root.findall(ns+'vsTrack')
    for track in tracks[1:]:root.remove(track)
    parts=tracks[0].findall(ns+'vsPart')
    for part in parts[1:]:tracks[0].remove(part)
    part=parts[0]
    part.find(ns+'t').text='0'
    for child in list(part):
        if child.tag in {ns+'note',ns+'mCtrl',ns+'cc'}:part.remove(child)
    for name in ('monoTrack','stTrack'):
        container=root.find(ns+name)
        if container is not None:
            for child in list(container):
                if child.tag==ns+'wavPart':container.remove(child)
    return tree

def export_score(score_path: Path, output: Path) -> dict:
    from .generated_vsqx import export_generated_score
    return export_generated_score(score_path, output)
