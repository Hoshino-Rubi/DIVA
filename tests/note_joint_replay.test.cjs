const test = require('node:test');
const assert = require('node:assert/strict');
const {matching, pitchLcs} = require('../tools/replay_note_pair_diagnostic.cjs');

const note = (start, end, midi) => ({start_sec:start,end_sec:end,pitch_midi:midi});

test('one-to-one matching does not double-count repeated pitch', () => {
  const reference = [note(1,1.2,60),note(1.2,1.4,60)];
  const output = [note(1,1.2,60),note(1,1.2,60)];
  assert.equal(matching(reference,output),1);
  assert.equal(pitchLcs(reference,output),2);
});

test('same pitch and onset with wrong offset is not joint agreement', () => {
  assert.equal(matching([note(1,1.2,60)],[note(1.02,1.3,60)],true),1);
  assert.equal(matching([note(1,1.2,60)],[note(1.02,1.3,60)]),0);
});

test('different MIDI can agree in boundaries without joint agreement', () => {
  assert.equal(matching([note(1,1.2,60)],[note(1,1.2,72)],false,false),1);
  assert.equal(matching([note(1,1.2,60)],[note(1,1.2,72)]),0);
});
