from __future__ import annotations
import copy
import json
from pathlib import Path
import threading
import unicodedata
import pytest
from any2xsampa.exporters.ipa_xsampa import IpaXSampaConverter
from any2xsampa.local.common import read, sha, write
from any2xsampa.local.engine import LocalEngine

EXPECTED = {
    'ɔ': 'O', 't͡sʰ': 't_s_h', 'ə': '@', 'ŋ': 'N', 'ɐ': '6',
    't͡s': 't_s', 'y': 'y', 'h': 'h', 'kʰ': 'k_h', 'sʰ': 's_h',
    'ɛ': 'E', 'f': 'f', 'ʊ': 'U', 'pʰ': 'p_h', 'tʰ': 't_h',
    'ɵ': '8', 'ø': '2', 'œ': '9',
}

@pytest.mark.parametrize('phone,expected', EXPECTED.items())
def test_common_real_ipa_is_not_missing(phone, expected):
    assert IpaXSampaConverter().convert(phone).xsampa == expected

@pytest.mark.parametrize('phone', ['ã', 'ë', 'ĭ', 'ṳ', 'ç', 'ç', 'ŋ̥', 't͡sʲː'])
def test_canonical_forms_agree_without_erasing_diacritics(phone):
    converter = IpaXSampaConverter()
    a = converter.convert(unicodedata.normalize('NFC', phone))
    b = converter.convert(unicodedata.normalize('NFD', phone))
    assert a.xsampa and a.xsampa == b.xsampa
    assert a.source_ipa == unicodedata.normalize('NFC', phone)

@pytest.mark.parametrize('phone', ['a☃', 'kˀ', '<blank>', '<unk>', '͡t', 't͡', 't͡͡s', 'ʰ', 'a b', 'UNKNOWN'])
def test_unknown_or_malformed_label_never_partially_converts(phone):
    result = IpaXSampaConverter().convert(phone)
    assert result.xsampa == '' and result.source_ipa == phone

def test_contrasts_ties_and_token_boundaries_survive_export():
    converter = IpaXSampaConverter()
    phones = ['t', 'tʰ', 't͡s', 't͡sʰ', 'ts', 'ɐ', 'a', 'ə', 'œ', 'ø', 'o', 'ŋ', 'n', 'm̩', 'm', 'aː']
    xsampa = converter.project(phones)['xsampa']
    assert len(set(xsampa)) == len(phones)
    assert converter.convert('t͡s').xsampa != ''.join(converter.project(['t', 's'])['xsampa'])
    assert converter.convert('m̩').xsampa == 'm='
    assert converter.convert('aː').xsampa == 'a:'

def test_statistics_count_occurrences_not_just_distinct_labels():
    p = IpaXSampaConverter().project(['a', 'kˀ', 'kˀ', '<unk>'])
    assert p['phone_count'] == 4
    assert p['unmapped_phone_count'] == 3 and p['unmapped_phone_ratio'] == .75
    assert p['unmapped_label_counts'] == {'<unk>': 1, 'kˀ': 2}
    assert p['recognition_accuracy_measured'] is False
    assert IpaXSampaConverter().project([])['unmapped_phone_ratio'] == 0

def old_result(tmp_path: Path) -> Path:
    file = tmp_path/'旧结果.json'
    write(file, {'schema_version':'local_task_result_v1', 'mode':'real',
        'segments':[{'normalized_ipa':p, 'xsampa':''} for p in EXPECTED]})
    return file

def test_preview_reexports_old_results_without_model_or_source_mutation(tmp_path, monkeypatch):
    file = old_result(tmp_path)
    original = file.read_bytes()
    from any2xsampa.local import engine
    def forbidden(*_): raise AssertionError('Preview must never load a model')
    monkeypatch.setattr(engine, 'WorkerClient', forbidden)
    p = LocalEngine().execute('preview_xsampa', {'result_path':str(file), 'result_sha256':sha(file)},
                             threading.Event(), lambda _:None)
    assert p['xsampa'] == list(EXPECTED.values())
    assert p['unmapped_phone_count'] == 0 and p['model_forwards'] == 0
    assert file.read_bytes() == original
    with pytest.raises(ValueError, match='version'):
        LocalEngine.preview_xsampa({'result_path':str(file), 'result_sha256':'bad'})

def test_machine_and_manual_exports_share_preview_mapping(tmp_path):
    file = old_result(tmp_path)
    original = file.read_bytes()
    out = tmp_path/'machine-xsampa.txt'
    LocalEngine.export({'result_path':str(file), 'output':str(out)})
    assert out.read_text(encoding='utf8').strip().split() == list(EXPECTED.values())
    proof = read(out.with_name(out.name+'.provenance.json'))
    assert proof['xsampa_export']['unmapped_phone_count'] == 0
    assert proof['xsampa_export']['mapping_sha256'] == IpaXSampaConverter().sha256
    edits = [{'index':i,'ipa':p} for i,p in enumerate(EXPECTED)]
    edits[0]['ipa'] = 'kˀ'
    LocalEngine.draft({'result_path':str(file), 'result_sha256':sha(file), 'edits':edits})
    out2 = tmp_path/'draft-xsampa.txt'
    LocalEngine.export({'result_path':str(file), 'output':str(out2), 'use_draft':True})
    assert out2.read_text(encoding='utf8').split()[0] == '[UNMAPPED:kˀ]'
    assert file.read_bytes() == original

def test_save_result_does_not_rewrite_worker_evidence_or_ipa(tmp_path):
    segments = [{'normalized_ipa':p, 'xsampa':'', 'start_sec':.1, 'end_sec':.2,
                 'metadata':{'raw_label_trace':p}} for p in EXPECTED]
    evidence = {'segments':segments, 'time_basis':'original_audio'}
    before = copy.deepcopy(evidence)
    result = LocalEngine.save_result(tmp_path, 'real', {'start_sec':0,'end_sec':1}, evidence)
    assert evidence == before
    assert [s['normalized_ipa'] for s in result['segments']] == list(EXPECTED)
    assert result['unmapped'] == []
    assert result['xsampa_export']['unmapped_phone_count'] == 0
    assert [s['metadata'] for s in result['segments']] == [s['metadata'] for s in segments]

