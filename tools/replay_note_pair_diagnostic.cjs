"use strict";

const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");

function hash(filename) {
  return crypto.createHash("sha256").update(fs.readFileSync(filename)).digest("hex");
}

function matching(reference, prediction, onsetOnly = false, requirePitch = true) {
  const links = reference.map(ref => prediction.flatMap((pred, index) =>
    Math.abs(ref.start_sec - pred.start_sec) <= .05 + 1e-9 &&
    (onsetOnly || Math.abs(ref.end_sec - pred.end_sec) <= .05 + 1e-9) &&
    (!requirePitch || ref.pitch_midi === pred.pitch_midi) ? [index] : []));
  const matched = Array(prediction.length).fill(-1);
  function visit(ref, seen) {
    for (const pred of links[ref]) {
      if (seen.has(pred)) continue;
      seen.add(pred);
      if (matched[pred] < 0 || visit(matched[pred], seen)) {
        matched[pred] = ref;
        return true;
      }
    }
    return false;
  }
  let count = 0;
  for (let ref = 0; ref < reference.length; ref++) if (visit(ref, new Set())) count++;
  return count;
}

function pitchLcs(reference, prediction) {
  let old = Array(prediction.length + 1).fill(0);
  for (const ref of reference) {
    const next = [0];
    prediction.forEach((pred, index) => next.push(ref.pitch_midi === pred.pitch_midi ?
      old[index] + 1 : Math.max(old[index + 1], next[index])));
    old = next;
  }
  return old[old.length - 1];
}

function replay(diagnostic, pairRun) {
  const rows = [];
  for (const row of diagnostic.rows) {
    const output = role => path.join(pairRun, "jobs", `${row.example_id}-${role}`, "prediction.json");
    const mixedFile = output("mixed"), vocalFile = output("vocal");
    if (hash(mixedFile) !== row.mixed_prediction_sha256 ||
        hash(vocalFile) !== row.vocal_prediction_sha256) throw new Error("Frozen prediction SHA mismatch");
    const mixed = JSON.parse(fs.readFileSync(mixedFile, "utf8")).notes;
    const vocal = JSON.parse(fs.readFileSync(vocalFile, "utf8")).notes;
    const computed = {
      mixed_notes: mixed.length, vocal_notes: vocal.length,
      pitch_sequence_lcs: pitchLcs(mixed, vocal),
      onset_only_agreements_50ms: matching(mixed, vocal, true, false),
      boundary_only_agreements_50ms: matching(mixed, vocal, false, false),
      pitch_onset_agreements_50ms: matching(mixed, vocal, true, true),
      joint_agreements_pitch_onset_offset_50ms: matching(mixed, vocal)
    };
    computed.mixed_without_joint_agreement = mixed.length - computed.joint_agreements_pitch_onset_offset_50ms;
    computed.vocal_without_joint_agreement = vocal.length - computed.joint_agreements_pitch_onset_offset_50ms;
    for (const [key, value] of Object.entries(computed)) {
      if (value !== row[key]) throw new Error(`${row.song_id}: ${key} differs (${value} vs ${row[key]})`);
    }
    rows.push({song_id: row.song_id, ...computed});
  }
  const totals = Object.fromEntries(Object.keys(rows[0]).filter(key => key !== "song_id")
    .map(key => [key, rows.reduce((total, row) => total + row[key], 0)]));
  for (const [key, value] of Object.entries(totals)) {
    if (diagnostic.totals[key] !== value) throw new Error(`Independent total differs: ${key}`);
  }
  return {schema_version:"game_pair_independent_replay_v1", original_truth_opened:false,
    model_forwards:0, formal_accuracy_measured:false, rows, totals};
}

if (require.main === module) {
  const [diagnosticPath, pairRun, outputPath] = process.argv.slice(2);
  if (!diagnosticPath || !pairRun || !outputPath) throw new Error("Usage: node replay_note_pair_diagnostic.cjs DIAGNOSTIC PAIR_RUN OUTPUT");
  const diagnostic = JSON.parse(fs.readFileSync(diagnosticPath, "utf8"));
  const result = replay(diagnostic, pairRun);
  result.diagnostic_sha256 = hash(diagnosticPath);
  fs.writeFileSync(outputPath, JSON.stringify(result, null, 2) + "\n", {flag:"wx"});
  process.stdout.write(`Independent replay: ${result.rows.length} pairs, ${result.totals.joint_agreements_pitch_onset_offset_50ms} agreements.\n`);
}

module.exports = {matching, pitchLcs, replay};
