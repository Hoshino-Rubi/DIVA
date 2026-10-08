(function (root) {
  "use strict";
  const SCHEMA = "original_note_review_v1";

  function fresh(packet, hash) {
    return {
      schema_version: SCHEMA,
      packet_sha256: hash,
      reviewer: "",
      raw_statement: "",
      clips: packet.clips.map(source => ({
        song_id: source.song_id,
        example_id: source.example_id,
        source_audio_sha256: source.source_audio_sha256,
        source_start_sample: source.source_start_sample,
        source_sample_rate: source.source_sample_rate,
        review_state: "pending",
        all_notes_reviewed: false,
        silent_confirmed: false,
        raw_statement: "",
        notes: [],
        uncertain_intervals: [],
        edit_log: [],
        next_id: 1
      }))
    };
  }

  function validateImport(packet, hash, answer) {
    if (!answer || answer.schema_version !== SCHEMA || answer.packet_sha256 !== hash ||
        !Array.isArray(answer.clips) || answer.clips.length !== packet.clips.length) {
      throw new Error("答案版本、来源哈希或五段数量不匹配");
    }
    answer.clips.forEach((clip, index) => {
      const source = packet.clips[index];
      for (const key of ["song_id", "example_id", "source_audio_sha256", "source_start_sample", "source_sample_rate"]) {
        if (clip[key] !== source[key]) throw new Error(`${source.song_id} 的 ${key} 不匹配`);
      }
      if (!Array.isArray(clip.notes) || !Array.isArray(clip.uncertain_intervals) ||
          !Array.isArray(clip.edit_log) || !Number.isInteger(clip.next_id) || clip.next_id < 1) {
        throw new Error(`${source.song_id} 的音符或操作记录损坏`);
      }
      const seen = new Set();
      clip.notes.forEach(note => {
        if (seen.has(note.id) || typeof note.id !== "string" ||
            !Number.isInteger(note.start_sample_abs) || !Number.isInteger(note.end_sample_abs) ||
            !Number.isInteger(note.pitch_midi) || note.pitch_midi < 0 || note.pitch_midi > 127 ||
            note.start_sample_abs < source.source_start_sample ||
            note.end_sample_abs > source.source_end_sample_exclusive ||
            note.start_sample_abs >= note.end_sample_abs) {
          throw new Error(`${source.song_id} 有非法音符`);
        }
        seen.add(note.id);
      });
    });
    if (typeof answer.reviewer !== "string" || typeof answer.raw_statement !== "string") {
      throw new Error("审核人或原话字段无效");
    }
    return answer;
  }

  function relativeSec(source, sampleAbs) {
    return (sampleAbs - source.source_start_sample) / source.source_sample_rate;
  }

  function absoluteSample(source, seconds) {
    const duration = (source.source_end_sample_exclusive - source.source_start_sample) / source.source_sample_rate;
    return source.source_start_sample + Math.round(Math.min(duration, Math.max(0, seconds)) * source.source_sample_rate);
  }

  function addNote(clip, source, startSec, endSec, pitchMidi) {
    const start = absoluteSample(source, startSec);
    const end = absoluteSample(source, endSec);
    if (end <= start || !Number.isInteger(pitchMidi) || pitchMidi < 0 || pitchMidi > 127) {
      throw new Error("音符起止点或 MIDI 无效");
    }
    const note = {
      id: `note-${String(clip.next_id++).padStart(4, "0")}`,
      start_sample_abs: start,
      end_sample_abs: end,
      pitch_midi: pitchMidi,
      edge_status: "fully_audible",
      review_state: "draft",
      raw_statement: "",
      overlap_reason: "",
      origin: "manual"
    };
    clip.notes.push(note);
    clip.silent_confirmed = false;
    clip.edit_log.push({action: "insert", note_id: note.id, at_utc: new Date().toISOString()});
    return note;
  }

  function record(clip, action, noteId) {
    clip.edit_log.push({action, note_id: noteId, at_utc: new Date().toISOString()});
  }

  function name(midi) {
    const labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    return `${labels[midi % 12]}${Math.floor(midi / 12) - 1}`;
  }

  const api = {SCHEMA, fresh, validateImport, relativeSec, absoluteSample, addNote, record, name};
  root.NoteReviewState = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window === "undefined" ? globalThis : window);
