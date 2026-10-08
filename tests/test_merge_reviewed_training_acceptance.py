from copy import deepcopy

import pytest

from tools.merge_reviewed_training_acceptance import merge_rows


def source(song='promise', start=0):
    return {'schema_version':'reviewed_training_data_acceptance_v1','rows':[
        {'song_id':song,'status':'data_accepted','split':'train','audio_role':'original',
         'audio_sha256':'a'*64,'start_sec':start,'end_sec':start+10,'reference_phones':37}]}


def test_three_songs_are_not_twenty_windows():
    result=merge_rows([source(s) for s in ('promise','freedom-flower','half-moon-night-song')])
    assert result['valid_windows']==3 and result['song_count']==3
    assert result['window_shortfall']==17 and result['song_shortfall']==0
    assert result['quantity_gate_passed'] is False and result['training_started'] is False


def test_same_retained_window_deduplicates_not_added_twice():
    row=source()
    result=merge_rows([row,deepcopy(row)])
    assert result['valid_windows']==1 and result['duplicate_rows_not_counted']==1


@pytest.mark.parametrize('kind',['overlap','conflict','version','rendered','pending','sealed','nonfinite'])
def test_invalid_cumulative_data_rejected(kind):
    first,second=source(),source(start=10)
    row=second['rows'][0]
    if kind=='overlap': row.update(start_sec=5,end_sec=15)
    elif kind=='conflict': row.update(start_sec=0,end_sec=10,reference_phones=38)
    elif kind=='version': row['audio_sha256']='b'*64
    elif kind=='rendered': row['audio_role']='rendered'
    elif kind=='pending': row['status']='pending'
    elif kind=='sealed': row['song_id']='king-of-karaoke'
    else: row['start_sec']=float('nan')
    with pytest.raises(ValueError): merge_rows([first,second])


def test_contiguous_windows_are_not_overlap():
    result=merge_rows([source(start=0),source(start=10)])
    assert result['valid_windows']==2 and result['unique_audio_seconds']==20
