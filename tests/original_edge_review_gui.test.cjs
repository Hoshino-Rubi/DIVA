'use strict';
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const S = require('../tools/original_edge_review_gui/state.js');
const packet = {schema_version: 'original_edge_review_packet_v1', packet_sha256: 'a'.repeat(64), items: ['start', 'end'].map((side, i) => ({
  review_id: 'B0' + (i + 1), side, cut_in_preview_sec: 1.25, duration_sec: 2.5,
  notes: [{note_id: 'n0', lyric: '你', phones: ['n', 'e', 'i'], mapping_status: 'resolved_proxy'},
    {note_id: 'unresolved', lyric: '嗯', phones: [], mapping_status: 'unresolved'}],
}))};
const answer = (side = 'start') => ({review_id: side === 'start' ? 'B01' : 'B02', status: 'confirmed', note_id: 'n0',
  phones: (side === 'start' ? [1, 2] : [0, 1]).map(phone_index => ({phone_index, edge_status: 'full'})),
  statement: '我确认身份与可听性', audible_identity_confirmed: true});
const payload = answers => ({schema_version: 'original_edge_review_answers_v1', packet_sha256: packet.packet_sha256, reviewer: '人工',
  exported_at: '2026-09-15T00:00:00Z', answers});
test('initial selections are empty, note changes reset phones and affirmation', () => {
  assert.deepEqual(S.emptyDraft(), {status: '', note_id: null, phones: [], statement: '', audible_identity_confirmed: false});
  const selected = S.selectNote(answer(), 'n0'); assert.deepEqual(selected.phones, []); assert.equal(selected.audible_identity_confirmed, false);
  assert.deepEqual(S.setPhone(selected, 2, 'truncated').phones, [{phone_index: 2, edge_status: 'truncated'}]);
  assert.deepEqual(S.setPhone(S.setPhone(selected, 2, 'full'), 2, '').phones, []);
});
test('training edge export is isolated from old dev envelopes and preserves absence', () => {
  const trainPacket = {...structuredClone(packet), schema_version: 'replacement_edge_review_packet_v1'};
  trainPacket.items.forEach((item, i) => { item.review_id = i ? 'L02-E' : 'L02-S'; });
  assert.equal(S.checkPacket(trainPacket), trainPacket);
  const a = {...answer('end'), review_id: 'L02-E', observation_schema: 'phone_audibility_v2',
    phones: [{phone_index: 0, edge_status: 'absent'}, {phone_index: 1, edge_status: 'full'}, {phone_index: 2, edge_status: 'partial'}]};
  const out = S.exportAnswers(trainPacket, [a], 'human');
  assert.equal(out.schema_version, 'replacement_edge_review_answers_v1');
  assert.equal(out.answers[0].phones[0].edge_status, 'absent');
  assert.throws(() => S.validateImport(packet, out), /格式/);
  assert.throws(() => S.validateImport(trainPacket, {...out, schema_version: 'original_edge_review_answers_v2'}), /格式/);
  assert.throws(() => S.exportAnswers(trainPacket, [{...a, observation_schema: undefined}], 'human'), /新版/);
});
test('start plays exactly the right half; end plays exactly the left half; preview is explicit', () => {
  assert.deepEqual(S.playbackRange(packet.items[0]), {offset: 1.25, duration: 1.25});
  assert.deepEqual(S.playbackRange(packet.items[1]), {offset: 0, duration: 1.25});
  assert.deepEqual(S.playbackRange(packet.items[0], true), {offset: 0, duration: 2.5});
});
test('valid suffix/prefix and answered subsets export and roundtrip', () => {
  for (const side of ['start', 'end']) {
    const a = answer(side); assert.equal(S.validateImport(packet, payload([a])).answers.length, 1);
    assert.deepEqual(S.exportAnswers(packet, [a], '人工', '2026-09-15T00:00:00Z'), {...payload([a]), schema_version: 'original_edge_review_answers_v2'});
  }
  assert.deepEqual(S.validateImport(packet, payload([])).answers, []);
});
test('rejects packet mismatch, unknown/repeated review IDs and schema mismatch', () => {
  assert.throws(() => S.validateImport(packet, {...payload([answer()]), packet_sha256: 'b'.repeat(64)}), /哈希/);
  assert.throws(() => S.validateImport(packet, payload([{...answer(), review_id: 'B99'}])), /未知/);
  assert.throws(() => S.validateImport(packet, payload([answer(), answer()])), /重复/);
  assert.throws(() => S.validateImport(packet, {...payload([]), schema_version: 'old'}), /格式/);
});
test('confirmed requires reviewer, resolved_proxy, nonempty phones, statement and explicit affirmation', () => {
  for (const a of [{...answer(), note_id: null}, {...answer(), note_id: 'unresolved'}, {...answer(), phones: []},
    {...answer(), statement: ''}, {...answer(), audible_identity_confirmed: false}, {...answer(), audible_identity_confirmed: 'true'}]) {
    assert.throws(() => S.validateAnswer(packet, a, '人工'));
  }
  assert.throws(() => S.validateAnswer(packet, answer(), '   '), /审核人/);
});
test('rejects out of range, duplicate, fractional and internally skipped phones', () => {
  for (const indices of [[-1, 2], [2, 3], [2, 2], [1.5, 2], [0, 2], [0, 1]]) {
    assert.throws(() => S.validateAnswer(packet, {...answer(), phones: indices.map(phone_index => ({phone_index, edge_status: 'full'}))}, '人工'));
  }
  assert.throws(() => S.validateAnswer(packet, {...answer('end'), phones: [{phone_index: 1, edge_status: 'full'}]}, '人工'), /前缀/);
});
test('only cutting-edge phone can be truncated', () => {
  for (const side of ['start', 'end']) {
    const a = answer(side), edge = side === 'start' ? 0 : 1, interior = 1 - edge;
    a.phones[edge].edge_status = 'truncated'; assert.doesNotThrow(() => S.validateAnswer(packet, a, '人工'));
    a.phones[interior].edge_status = 'truncated'; assert.throws(() => S.validateAnswer(packet, a, '人工'), /切点/);
  }
});
test('uncertain and wrong correspondence carry no identity assertion or selected phones', () => {
  for (const status of ['uncertain', 'wrong_correspondence']) {
    const committed = S.commitDraft(packet, packet.items[0], {...answer(), status}, '');
    assert.equal(committed.note_id, null); assert.deepEqual(committed.phones, []); assert.equal(committed.audible_identity_confirmed, false);
    assert.throws(() => S.validateAnswer(packet, {...committed, phones: [{phone_index: 0, edge_status: 'full'}]}, ''), /不能携带/);
    assert.throws(() => S.validateAnswer(packet, {...committed, audible_identity_confirmed: true}, ''), /不能携带/);
  }
});
test('confirmation statement covers audibility without inferring absence causes', () => {
  const a = S.commitDraft(packet, packet.items[0], {...answer(), statement: '<img src=x onerror=alert(1)>'}, '人工');
  assert.match(a.statement, /首字/); assert.match(a.statement, /不据此推断缺失原因/);
  assert.ok(a.statement.includes('<img src=x onerror=alert(1)>'));
});
test('packet guards max six, duplicate IDs and exact center', () => {
  assert.equal(S.checkPacket(packet), packet);
  assert.throws(() => S.checkPacket({...packet, items: [...packet.items, packet.items[0]]}), /重复/);
  assert.throws(() => S.checkPacket({...packet, items: Array(7).fill(packet.items[0])}), /1–6/);
  assert.throws(() => S.checkPacket({...packet, items: [{...packet.items[0], cut_in_preview_sec: 0}]}), /1.25/);
});
test('offline UI has no fetch, innerHTML, model prediction UI or implicit audio autoplay', () => {
  const app = fs.readFileSync(path.join(__dirname, '../tools/original_edge_review_gui/app.js'), 'utf8');
  assert.doesNotMatch(app, /innerHTML|fetch|autoplay|setInterval/);
  assert.match(app, /textContent/); assert.ok(app.includes('source.start(0, range.offset, range.duration)'));
  assert.match(app, /previous_audit/); assert.match(app, /packet_sha256/);
});
function mount(storage = new Map()) {
  const viewport = {scrollX: 0, scrollY: 0};
  class Node {
    constructor(tag = '') { this.tag = tag; this.children = []; this.events = {}; this.textContent = ''; this.value = ''; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; viewport.scrollY = 0; }
    addEventListener(name, handler) { this.events[name] = handler; }
    setAttribute() {}
    getContext() { return {clearRect() {}, fillRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {}}; }
    click() { return this.events.click?.(); }
    remove() {}
  }
  const guiPacket = structuredClone(packet);
  guiPacket.items.forEach(item => { item.media = [{role: 'original', audio_file: 'media/test.wav'}]; item.previous_audit = {decision: 'corresponds', final_audit_response: {comment: '<script>not executed</script>'}}; });
  const ids = Object.fromEntries(['packet', 'audio-data', 'reviewer', 'notice', 'items', 'review', 'save', 'export', 'import'].map(id => [id, new Node()]));
  ids.packet.textContent = JSON.stringify(guiPacket); ids['audio-data'].textContent = JSON.stringify({'media/test.wav': 'AA=='});
  const starts = [], blobs = [];
  class AudioContext {
    async decodeAudioData() { return {duration: 2.5, getChannelData() { return new Float32Array(100); }}; }
    async resume() {}
    createBufferSource() { return {connect() {}, stop() {}, start(...args) { starts.push(args); }}; }
  }
  const sandbox = {window: Object.assign(viewport, {EdgeReviewState: S, AudioContext, confirm: () => true, scrollTo({left, top}) { this.scrollX = left; this.scrollY = top; }}),
    document: {getElementById: id => ids[id], createElement: tag => new Node(tag), createTextNode: text => ({textContent: text}), body: new Node()},
    localStorage: {getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value)},
    structuredClone, Uint8Array, atob, setTimeout() {},
    Blob: class {constructor(parts) { blobs.push(JSON.parse(parts[0])); }}, URL: {createObjectURL: () => 'blob:test', revokeObjectURL() {}}};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../tools/original_edge_review_gui/app.js'), 'utf8'), sandbox);
  function flatten(node) { return [node, ...(node.children || []).flatMap(flatten)]; }
  ids.items.replaceChildren = function (...children) { this.children = children; };
  const nodes = () => flatten(ids.review);
  return {ids, nodes, starts, blobs, storage, viewport, find: text => nodes().find(node => node.textContent === text)};
}
test('DOM event smoke: blank initial selections, safe prior text, exact half playback and uncertain submit/export', async () => {
  const gui = mount();
  assert.ok(gui.find('尚未选择 note；没有默认选中的音素。'));
  assert.ok(gui.find('<script>not executed</script>'));
  await gui.find('播放片段内侧 · 仅中线右侧').click();
  assert.deepEqual(gui.starts, [[0, 1.25, 1.25]]);
  const status = gui.nodes().find(node => node.tag === 'select'); status.value = 'uncertain'; status.events.change();
  gui.find('提交本项并下一项').click();
  assert.ok(gui.find('B02 · 结尾末字 / 末音素'));
  await gui.find('播放片段内侧 · 仅中线左侧').click();
  assert.deepEqual(gui.starts[1], [0, 0, 1.25]);
  gui.ids.export.click(); assert.equal(gui.blobs[0].answers[0].status, 'uncertain');
  assert.equal(gui.blobs[0].answers[0].note_id, null); assert.deepEqual(gui.blobs[0].answers[0].phones, []);
  const restored = mount(gui.storage); restored.ids.export.click(); assert.equal(restored.blobs[0].answers.length, 1);
});
test('DOM confirmed path requires manual note, phone, reviewer and affirmative action', () => {
  const gui = mount(); gui.ids.reviewer.value = '人工';
  const noteButton = gui.nodes().find(node => node.tag === 'button' && node.children.some(c => c.textContent === '你 · /n e i/'));
  noteButton.click();
  const fullButtons = gui.nodes().filter(node => node.tag === 'button' && node.textContent === '完整可听');
  fullButtons[2].click();
  const status = gui.nodes().find(node => node.tag === 'select'); status.value = 'confirmed'; status.events.change();
  gui.find('提交本项并下一项').click(); assert.match(gui.ids.notice.textContent, /明确勾选/);
  const checkbox = gui.nodes().find(node => node.type === 'checkbox'); checkbox.checked = true; checkbox.events.change();
  gui.find('提交本项并下一项').click(); gui.ids.export.click();
  assert.equal(gui.blobs[0].answers[0].status, 'confirmed');
  assert.deepEqual(gui.blobs[0].answers[0].phones, [{phone_index: 2, edge_status: 'full'}]);
});

test('audibility v2 accepts absent initial phone at end edge, without treating it as partial', () => {
  const draft = {...answer('end'), phones: [
    {phone_index: 0, edge_status: 'absent'},
    {phone_index: 1, edge_status: 'full'},
    {phone_index: 2, edge_status: 'full'}]};
  const a = S.commitDraft(packet, packet.items[1], draft, '人工');
  assert.equal(a.observation_schema, 'phone_audibility_v2');
  assert.deepEqual(a.phones, draft.phones);
  const exported = S.exportAnswers(packet, [a], '人工');
  assert.deepEqual(S.validateImport(packet, exported).answers[0], a);
  for (const status of ['partial', 'uncertain']) {
    assert.equal(S.validateAnswer(packet, {...a, phones: [{phone_index: 0, edge_status: status}]}, '人工').phones[0].edge_status, status);
  }
  assert.throws(() => S.validateImport(packet, {...exported, schema_version: 'original_edge_review_answers_v1'}), /旧格式/);
});
test('legacy truncation remains partial on explicit editing, never becomes absence', () => {
  const legacy = answer(); legacy.phones[0].edge_status = 'truncated';
  assert.equal(S.exportAnswers(packet, [legacy], '人工').answers[0].phones[0].edge_status, 'truncated');
  const edited = S.setPhone(legacy, 2, 'full');
  assert.equal(edited.phones[0].edge_status, 'partial');
  assert.ok(edited.phones.every(p => p.edge_status !== 'absent'));
});
test('DOM phoneme clicks retain the form nodes and scroll; bulk rerenders restore position', () => {
  const gui = mount(); gui.ids.reviewer.value = '人工';
  const note = gui.nodes().find(n => n.tag === 'button' && n.children.some(c => c.textContent === '你 · /n e i/'));
  gui.viewport.scrollY = 950; note.click();
  assert.equal(gui.viewport.scrollY, 950);
  const form = gui.ids.review.children.at(-1);
  gui.viewport.scrollY = 1250;
  const absent = gui.nodes().filter(n => n.tag === 'button' && n.textContent === '完全听不到');
  absent[0].click();
  assert.equal(gui.viewport.scrollY, 1250);
  assert.equal(gui.ids.review.children.at(-1), form);
  assert.equal(absent[0].className, 'active');
  const full = gui.nodes().filter(n => n.tag === 'button' && n.textContent === '完整可听');
  full[1].click(); full[2].click();
  assert.equal(gui.viewport.scrollY, 1250);
  const status = gui.nodes().find(n => n.tag === 'select'); status.value = 'confirmed'; status.events.change();
  assert.equal(gui.viewport.scrollY, 1250);
  const checkbox = gui.nodes().find(n => n.type === 'checkbox'); checkbox.checked = true; checkbox.events.change();
  gui.find('提交本项并下一项').click(); gui.ids.export.click();
  assert.deepEqual(gui.blobs[0].answers[0].phones.map(p => p.edge_status), ['absent', 'full', 'full']);
  const restored = mount(gui.storage); restored.ids.export.click();
  assert.deepEqual(restored.blobs[0].answers, gui.blobs[0].answers);
  restored.viewport.scrollY = 1000;
  restored.ids.items.children[0].click();
  restored.find('全部可听（显式选择全部为完整）').click();
  assert.equal(restored.viewport.scrollY, 1000);
});
