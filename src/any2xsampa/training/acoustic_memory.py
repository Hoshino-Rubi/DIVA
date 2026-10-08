"""Fixed nonparametric acoustic memory. No model loading or target-aware query."""
from __future__ import annotations

from typing import Any, Callable, Sequence
import numpy as np

from any2xsampa.training.ctc_position_bridge import position_occupancy

SONGS = ('freedom-flower', 'half-moon-night-song', 'like-you', 'promise', 'queen-road-east')
QUERY_FIELDS = frozenset(('example_id', 'song_id', 'split', 'audio_role', 'audio_path',
    'audio_sha256', 'start_sec', 'end_sec', 'architecture_sha256', 'frames', 'path',
    'sha256', 'audio_metadata', 'crop', 'geometry'))
META_FIELDS = frozenset(('sha256', 'sample_rate', 'channels', 'sample_width_bytes', 'frame_count', 'duration_sec'))
CROP_FIELDS = frozenset(('requested_start_sec', 'requested_end_sec', 'source_start_sample',
    'source_end_sample_exclusive', 'effective_start_sec', 'effective_end_sec', 'model_sample_rate',
    'model_samples', 'channel_policy', 'resample_policy', 'waveform_normalization'))


def validate_query(row: dict) -> None:
    """Whitelist, including nested fields: no reference/lyrics/notes/pitch fields."""
    if set(row) != QUERY_FIELDS or set(row['audio_metadata']) != META_FIELDS:
        raise ValueError('query fields must be audio/cache identity only')
    if set(row['crop']) != CROP_FIELDS or set(row['geometry']) != {'frame_count', 'stride_samples', 'receptive_field_samples'}:
        raise ValueError('query nested fields contain unsupported conditions')
    if row['split'] != 'train' or row['song_id'] not in SONGS or row['audio_role'] not in ('original', 'rendered'):
        raise ValueError('query song/role leakage')
    if row['audio_metadata']['sha256'] != row['audio_sha256']:
        raise ValueError('audio hash identity mismatch')
    if row['frames'] != row['geometry']['frame_count'] or row['frames'] < 1:
        raise ValueError('frame geometry mismatch')
    crop = row['crop']; rate = row['audio_metadata']['sample_rate']
    if rate <= 0 or crop['model_sample_rate'] != 16000:
        raise ValueError('sample rate mismatch')
    for edge in ('start', 'end'):
        value = float(row[edge + '_sec'])
        if not np.isfinite(value) or value < 0 or crop['requested_' + edge + '_sec'] != value:
            raise ValueError('invalid requested sample offset')
        sample = crop['source_' + edge + '_sample' + ('_exclusive' if edge == 'end' else '')]
        if sample != round(value * rate) or abs(crop['effective_' + edge + '_sec'] - sample / rate) > 1e-9:
            raise ValueError('sample offset mismatch')
    if not 0 < row['end_sec'] - row['start_sec'] <= 20.000001:
        raise ValueError('invalid query duration')


def validate_sources(rows: Sequence[dict], held: str) -> None:
    if held not in SONGS or not rows or len({r['example_id'] for r in rows}) != len(rows):
        raise ValueError('invalid source identity')
    for row in rows:
        if set(row) != {'query', 'target_ids', 'reference_sha256', 'reference_nature', 'example_id', 'song_id'}:
            raise ValueError('invalid source schema')
        q = row['query']; validate_query(q)
        if q['audio_role'] != 'original' or q['song_id'] == held or row['song_id'] != q['song_id'] or row['example_id'] != q['example_id']:
            raise ValueError('held song/role leaked into bank')
        target = row['target_ids']
        if not target or any(type(i) is not int or not 0 < i < 428 for i in target):
            raise ValueError('invalid source labels')
        if len(target) + sum(a == b for a, b in zip(target, target[1:])) > q['frames']:
            raise ValueError('insufficient CTC source frames')


def normalized_keys(values: Any) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 1 or not np.isfinite(a).all():
        raise ValueError('invalid memory features')
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    if (norms <= 0).any() or not np.isfinite(norms).all():
        raise ValueError('zero/nonfinite feature norm')
    return a / norms


def soft_ctc_values(logits: Any, target: Sequence[int]) -> tuple[np.ndarray, dict]:
    result = position_occupancy(logits, target)
    states = result['state_posteriors']
    labels = np.zeros(2 * len(target) + 1, dtype=np.int64); labels[1::2] = target
    gamma = np.zeros((len(states), np.asarray(logits).shape[1]), dtype=np.float64)
    for state, label in enumerate(labels):
        gamma[:, label] += states[:, state]
    probabilities(gamma)
    return gamma, {'log_target_probability': result['log_target_probability'],
        'reference_conditioned': True, 'acoustic_boundary_truth': False,
        'maximum_row_sum_error': float(abs(gamma.sum(1) - 1).max())}


def probabilities(value: Any) -> np.ndarray:
    p = np.asarray(value, dtype=np.float64)
    if p.ndim != 2 or min(p.shape) < 1 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('invalid probability array')
    if not np.allclose(p.sum(1), 1, atol=1e-7, rtol=0):
        raise ValueError('probability rows not normalized')
    return p


def softmax(logits: Any) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError('invalid logits')
    p = np.exp(x - x.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    return probabilities(p)


def squared_distances(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
    # General squared Euclidean expression, not an approximate index.
    d = (query * query).sum(1)[:, None] + (keys * keys).sum(1)[None, :] - 2 * (query @ keys.T)
    if not np.isfinite(d).all() or d.min() < -1e-10:
        raise ValueError('invalid squared distances')
    return np.maximum(d, 0.)  # roundoff only, not a fallback distance


def exact_neighbors(query: np.ndarray, keys: np.ndarray, *, k: int = 16,
                    query_songs: np.ndarray | None = None, key_songs: np.ndarray | None = None,
                    block: int = 128, source_block: int = 2048,
                    check: Callable[[], None] = lambda: None) -> tuple[np.ndarray, np.ndarray]:
    if k < 1 or block < 1 or source_block < 1 or len(query) < 1 or len(keys) < 1:
        raise ValueError('empty/invalid neighbor request')
    if query.ndim != 2 or keys.ndim != 2 or query.shape[1] != keys.shape[1] or not np.isfinite(query).all() or not np.isfinite(keys).all():
        raise ValueError('invalid neighbor features')
    if (query_songs is None) != (key_songs is None):
        raise ValueError('incomplete song exclusion')
    if query_songs is not None and (len(query_songs) != len(query) or len(key_songs) != len(keys)):
        raise ValueError('song exclusion dimensions')
    count = min(k, len(keys)); all_i = []; all_d = []
    for start in range(0, len(query), block):
        q = query[start:start + block]; ii = np.empty((len(q), 0), dtype=np.int64); dd = np.empty((len(q), 0))
        for offset in range(0, len(keys), source_block):
            check(); part = keys[offset:offset + source_block]; d = squared_distances(q, part)
            if query_songs is not None:
                d[query_songs[start:start + len(q), None] == key_songs[None, offset:offset + len(part)]] = np.inf
            ids = np.broadcast_to(np.arange(offset, offset + len(part)), d.shape)
            joined_d = np.concatenate((dd, d), axis=1); joined_i = np.concatenate((ii, ids), axis=1)
            order = np.lexsort((joined_i, joined_d), axis=1)[:, :count]
            dd = np.take_along_axis(joined_d, order, axis=1); ii = np.take_along_axis(joined_i, order, axis=1)
        if not np.isfinite(dd).all():
            raise ValueError('no cross-song memory neighbor')
        all_i.append(ii); all_d.append(dd)
    return np.concatenate(all_i), np.concatenate(all_d)


def memory_temperature(keys: np.ndarray, songs: np.ndarray, check: Callable[[], None] = lambda: None) -> tuple[float, np.ndarray]:
    if len(set(songs.tolist())) < 2:
        raise ValueError('temperature requires other training songs')
    _, d = exact_neighbors(keys, keys, k=1, query_songs=songs, key_songs=songs, check=check)
    return max(1e-6, float(np.median(d))), d[:, 0]


def shuffled_values(values: np.ndarray, spans: Sequence[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(17); permutation = np.arange(len(values)); expected = 0
    for start, end in spans:
        if start != expected or end <= start or end > len(values):
            raise ValueError('noncontiguous shuffle windows')
        permutation[start:end] = rng.permutation(np.arange(start, end)); expected = end
    if expected != len(values):
        raise ValueError('shuffle windows incomplete')
    return values[permutation], permutation


def memory_distribution(values: np.ndarray, indices: np.ndarray, distances: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0 or indices.shape != distances.shape or (distances < 0).any():
        raise ValueError('invalid retrieval weights')
    if not np.isfinite(distances).all() or indices.dtype.kind not in 'iu' or (indices < 0).any() or (indices >= len(values)).any():
        raise ValueError('invalid neighbor indices')
    weights = softmax(-distances / temperature)
    output = np.einsum('tk,tkc->tc', weights, values[indices], optimize=True)
    return probabilities(output)


def fuse(baseline: np.ndarray, memory: np.ndarray, weight: float = .25) -> np.ndarray:
    probabilities(baseline); probabilities(memory)
    if baseline.shape != memory.shape or not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('invalid fusion')
    if weight == 0:
        return baseline.copy()
    return probabilities((1 - weight) * baseline + weight * memory)
