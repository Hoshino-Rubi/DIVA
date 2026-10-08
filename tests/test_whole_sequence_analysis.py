from copy import deepcopy
import pytest
from any2xsampa.training.whole_sequence_analysis import compare_systems,source_coverage,scalar_distance


def test_independent_edits_keep_repeats_and_insertions():
    assert scalar_distance(['i','i'],['i'])==1
    assert scalar_distance(['kʰ','i'],['k','i','n'])==2
    assert scalar_distance([],['h'])==1


def test_paired_comparison_rejects_reference_changes_and_counts_errors():
    base=[{'example_id':'one','song_id':'freedom-flower','reference':['kʰ','i'],'prediction':['k','i']}]
    improved=deepcopy(base);improved[0]['prediction']=['kʰ','i']
    result=compare_systems({'before':base,'after':improved})
    assert result['systems']['before']['edit_distance']==1
    assert result['systems']['after']['edit_distance']==0
    assert result['independent_scalar_distances_verified']==2
    changed=deepcopy(improved);changed[0]['reference']=['k','i']
    with pytest.raises(ValueError,match='reference identity'):
        compare_systems({'before':base,'after':changed})


def test_unique_source_coverage_counts_full_and_overlapping_windows_once():
    common={'song_id':'freedom-flower','audio_sha256':'a'*64,'split':'train','audio_role':'original'}
    short=[{**common,'start_sec':10.,'end_sec':20.},{**common,'start_sec':20.,'end_sec':30.},
           {**common,'song_id':'promise','audio_sha256':'b'*64,'start_sec':0.,'end_sec':10.}]
    result=source_coverage(short,[{'song_id':'freedom-flower','audio_sha256':'a'*64,'duration_sec':40.}])
    assert result['previous_unique_original_seconds']==30.
    assert result['unique_original_seconds']==50.
    assert result['additional_unique_original_seconds']==20.
    with pytest.raises(ValueError,match='versions differ'):
        source_coverage(short,[{'song_id':'freedom-flower','audio_sha256':'c'*64,'duration_sec':40.}])
