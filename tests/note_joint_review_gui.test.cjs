const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
const state = require('../tools/note_joint_review_gui/state.js');
const template = fs.readFileSync(path.join(root, 'tools/note_joint_review_gui/index.html'), 'utf8');
const app = fs.readFileSync(path.join(root, 'tools/note_joint_review_gui/app.js'), 'utf8');
const source = {song_id:'freedom-flower',example_id:'batch-r05',source_audio_sha256:'a'.repeat(64),
  source_start_sample:2205000,source_end_sample_exclusive:2646000,source_sample_rate:44100};
const packet = {clips:[source]};
const hash = 'b'.repeat(64);

test('empty packet never prefills reference pitches or model predictions', () => {
  const answer = state.fresh(packet,hash);
  assert.equal(answer.clips[0].review_state,'pending');
  assert.deepEqual(answer.clips[0].notes,[]);
  assert.ok(!template.includes('prediction.json'));
  assert.ok(!app.includes('prediction.json'));
  assert.ok(!template.includes('VSQX'));
});

test('sample conversion follows original audio source clock', () => {
  assert.equal(state.absoluteSample(source, 0), 2205000);
  assert.equal(state.absoluteSample(source, 10), 2646000);
  assert.equal(state.absoluteSample(source, .5), 2227050);
  assert.equal(state.relativeSec(source, 2227050), .5);
});

test('insertion and duplicate pitch remain separate manual events', () => {
  const answer = state.fresh(packet,hash);
  const a = state.addNote(answer.clips[0],source,1,1.4,60);
  const b = state.addNote(answer.clips[0],source,1.4,1.8,60);
  assert.notEqual(a.id,b.id);
  assert.equal(answer.clips[0].edit_log.length,2);
  assert.equal(a.origin,'manual');
  assert.equal(a.review_state,'draft');
});

test('packet mismatch and tampered source reject import', () => {
  const answer = state.fresh(packet,hash);
  assert.throws(() => state.validateImport(packet,'c'.repeat(64),answer), /哈希/);
  answer.clips[0].source_start_sample++;
  assert.throws(() => state.validateImport(packet,hash,answer), /不匹配/);
});

test('invalid pitch or out-of-window note reject import', () => {
  const answer = state.fresh(packet,hash);
  state.addNote(answer.clips[0],source,1,2,60).pitch_midi=128;
  assert.throws(() => state.validateImport(packet,hash,answer), /非法音符/);
  answer.clips[0].notes[0].pitch_midi=60;
  answer.clips[0].notes[0].end_sample_abs=2646001;
  assert.throws(() => state.validateImport(packet,hash,answer), /非法音符/);
});

test('page is offline and keeps the five song navigation', () => {
  assert.match(template,/connect-src 'none'/);
  assert.match(template,/id="songs"/);
  assert.match(template,/id="export"/);
  assert.match(template,/id="import"/);
  assert.match(template,/id="wave"/);
  assert.match(template,/id="piano"/);
  assert.match(app,/localStorage/);
  assert.match(app,/textContent/);
});
