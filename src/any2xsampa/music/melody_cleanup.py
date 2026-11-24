"""Post-stitch, monophonic GAME fragment cleanup on the score's actual tick grid."""
from __future__ import annotations


def clean_melody(pitch: dict, ticks_per_second: float, *, v5: bool = False, v6: bool = False) -> dict:
    notes = [dict(row) for row in pitch["note_candidates"]]
    def duration(row):
        return row["duration_sec"]
    def end(row):
        return row["start_sec"] + duration(row)
    def ticks(row):
        return round(end(row)*ticks_per_second) - round(row["start_sec"]*ticks_per_second)
    def short(row):
        if v6:
            return duration(row) < .065 - 1e-9
        return (duration(row) < .075 - 1e-9 or
                ticks(row) < max(80,.075*ticks_per_second-1e-9))
    def adjacent(i):
        return [j for j in (i-1, i+1) if 0 <= j < len(notes)]
    def spike(i):
        neighbors = adjacent(i)
        return (min(duration(notes[i]),ticks(notes[i])/ticks_per_second) < .120 - 1e-9 and bool(neighbors)
                and all(abs(notes[i]["midi"] - notes[j]["midi"]) > 7 for j in neighbors))
    def gap(i, j):
        a, b = sorted((i, j))
        return notes[b]["start_sec"] - end(notes[a])
    audit = dict(input=len(notes), short_detected=sum(map(short, notes)),
                 spikes_detected=sum(map(spike, range(len(notes)))),
                 short_jump_75=sum(duration(n)<.075-1e-9 and spike(i) for i,n in enumerate(notes)), short_merged=0,
                 short_dropped=0, spikes_snapped=0)
    # Only a stable note can absorb a fragment. Never extend across a phrase rest.
    while any(map(short, notes)):
        indices = [i for i,r in enumerate(notes) if short(r)]
        matches = [(gap(i,j), abs(notes[i]["midi"]-notes[j]["midi"]), -duration(notes[j]), i,j)
                   for i in indices for j in adjacent(i) if not short(notes[j]) and gap(i,j) < .04-1e-9]
        if not matches:
            if v6:  # No neighbouring host: keep the isolated event, do not invent a merge.
                break
            notes = [r for r in notes if not short(r)]
            audit["short_dropped"] += len(indices)
            break
        *_, i,j = min(matches)
        first = min(notes[i]["start_sec"], notes[j]["start_sec"])
        last = max(end(notes[i]), end(notes[j]))
        notes[j].update(start_sec=first, duration_sec=last-first)
        del notes[i]
        audit["short_merged"] += 1
    if v6:
        pass  # No interval- or duration-based pitch flattening in V6.
    elif v5:
        from .singing_v5_melody import repair
        events = repair(notes,ticks_per_second)
        audit.update(pitch_repairs=events,spikes_snapped=sum(e["old"]!=e["new"] for e in events),spikes_detected=len(events))
    else:
        while (i := next((j for j in range(len(notes)) if spike(j)), None)) is not None:
            j = min(adjacent(i), key=lambda j: (-duration(notes[j]),gap(i,j)))
            notes[i]["midi"] = notes[j]["midi"]
            audit["spikes_snapped"] += 1
    audit.update(output=len(notes), remaining_short=sum(map(short,notes)),
                 remaining_spikes=sum(map(spike,range(len(notes)))))
    return {"note_candidates": notes, "melody_cleanup": audit}


def game_rows(pitch: dict, duration: float) -> list[dict]:
    """Preserve the monophonic GAME clock; one recognized syllable per note."""
    from math import isfinite
    rows = []
    for item in pitch["note_candidates"]:
        start = float(item["start_sec"])
        end = min(duration, start + item["duration_sec"])
        if not 0 <= start < end or (rows and start < rows[-1]["end_sec"] - 1e-8):
            raise ValueError("GAME note outside the audio or overlapping")
        midi = item["midi"]
        if not isfinite(midi) or not 0 <= midi <= 127 or int(midi) != midi:
            raise ValueError("Invalid GAME pitch")
        rows.append(dict(index=len(rows), start_sec=start, end_sec=end, duration_sec=end-start,
                         pitch=int(midi), pitch_source="GAME"))
    if not rows:
        raise ValueError("No melody notes")
    return rows
