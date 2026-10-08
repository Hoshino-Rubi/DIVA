(function (root) {
  'use strict';
  const SCHEMA = 'original_edge_review_answers_v2';
  const OBSERVATION = 'phone_audibility_v2';
  const TRAIN_PACKET = 'replacement_edge_review_packet_v1';
  const TRAIN_ANSWER = 'replacement_edge_review_answers_v1';
  function demand(test, message) { if (!test) throw new Error(message); }
  function checkPacket(packet) {
    demand(packet && ['original_edge_review_packet_v1', TRAIN_PACKET].includes(packet.schema_version), '问卷格式不匹配');
    demand(typeof packet.packet_sha256 === 'string' && /^[a-f0-9]{64}$/.test(packet.packet_sha256), '缺少有效 packet_sha256');
    demand(Array.isArray(packet.items) && packet.items.length > 0 && packet.items.length <= 6, '问卷必须有 1–6 项');
    const ids = new Set();
    packet.items.forEach(item => {
      demand(typeof item.review_id === 'string' && !ids.has(item.review_id), '问卷 review_id 重复或无效');
      ids.add(item.review_id);
      demand(['start', 'end'].includes(item.side), '未知切点类型');
      demand(item.cut_in_preview_sec === 1.25 && item.duration_sec === 2.5, '问卷预览中线必须为 1.25 秒');
      demand(Array.isArray(item.notes), '缺少 note 列表');
      const notes = new Set();
      item.notes.forEach(note => {
        demand(typeof note.note_id === 'string' && !notes.has(note.note_id), 'note_id 重复或无效');
        notes.add(note.note_id);
        demand(Array.isArray(note.phones) && note.phones.every(p => typeof p === 'string' && p.length > 0), 'IPA phones 格式错误');
      });
    });
    return packet;
  }
  function emptyDraft() {
    return {status: '', note_id: null, phones: [], statement: '', audible_identity_confirmed: false};
  }
  function selectNote(draft, noteId) {
    return {...draft, observation_schema: OBSERVATION, note_id: noteId, phones: [], audible_identity_confirmed: false};
  }
  function setPhone(draft, index, edgeStatus) {
    const phones = draft.phones.filter(p => p.phone_index !== index).map(p => ({...p, edge_status: p.edge_status === 'truncated' ? 'partial' : p.edge_status}));
    if (edgeStatus) phones.push({phone_index: index, edge_status: edgeStatus});
    phones.sort((a, b) => a.phone_index - b.phone_index);
    return {...draft, observation_schema: OBSERVATION, phones, audible_identity_confirmed: false};
  }
  function validateAnswer(packet, answer, reviewer) {
    demand(answer && typeof answer === 'object', '答案必须为对象');
    demand(answer.observation_schema === undefined || answer.observation_schema === OBSERVATION, '未知听觉记录版本');
    const observation = answer.observation_schema === OBSERVATION;
    const item = packet.items.find(row => row.review_id === answer.review_id);
    demand(item, '未知 review_id');
    demand(['confirmed', 'uncertain', 'wrong_correspondence'].includes(answer.status), '请明确选择确认或不确定');
    demand(typeof answer.statement === 'string' && answer.statement.trim().length > 0, '缺少确认语句或说明');
    demand(Array.isArray(answer.phones), 'phones 必须为列表');
    if (answer.status !== 'confirmed') {
      demand(answer.note_id === null && answer.phones.length === 0 && answer.audible_identity_confirmed === false, '不确定或对应有误不能携带音素确认');
      return structuredClone(answer);
    }
    demand(typeof reviewer === 'string' && reviewer.trim().length > 0, '请填写审核人');
    const note = item.notes.find(row => row.note_id === answer.note_id);
    demand(note && note.mapping_status === 'resolved_proxy', '请选择 IPA 代理映射为 resolved_proxy 的 note');
    demand(answer.audible_identity_confirmed === true, '必须明确勾选身份与可听性确认');
    demand(answer.phones.length > 0, '请选择实际听到的 IPA，不能默认全选');
    const seen = new Set();
    answer.phones.forEach(p => {
      demand(p && Number.isInteger(p.phone_index) && p.phone_index >= 0 && p.phone_index < note.phones.length, 'phone_index 越界');
      demand(!seen.has(p.phone_index), 'phone_index 重复'); seen.add(p.phone_index);
      demand((observation ? ['full', 'partial', 'absent', 'uncertain'] : ['full', 'truncated']).includes(p.edge_status), '音素状态与记录版本不匹配');
    });
    if (observation) return {...structuredClone(answer), phones: [...answer.phones].sort((a, b) => a.phone_index - b.phone_index)};
    const indices = [...seen].sort((a, b) => a - b);
    demand(indices.every((x, i) => i === 0 || x === indices[i - 1] + 1), '音素不能内部跳选');
    demand(item.side === 'start' ? indices.at(-1) === note.phones.length - 1 : indices[0] === 0,
      item.side === 'start' ? 'start 必须为所选 note 的连续后缀' : 'end 必须为所选 note 的连续前缀');
    const edgeIndex = item.side === 'start' ? indices[0] : indices.at(-1);
    demand(answer.phones.every(p => p.edge_status !== 'truncated' || p.phone_index === edgeIndex), '只有紧邻正式切点的音素可以标为截断');
    return {...structuredClone(answer), phones: [...answer.phones].sort((a, b) => a.phone_index - b.phone_index)};
  }
  function commitDraft(packet, item, draft, reviewer) {
    const confirmed = draft.status === 'confirmed';
    const prefix = confirmed ? '我明确确认：所选字是中线处片段内的' + (item.side === 'start' ? '首字' : '末字') + '，我按听觉记录各音素完整可听、部分可听、完全听不到或不确定；未选择表示尚未判断，不据此推断缺失原因。' :
      draft.status === 'uncertain' ? '我不确定中线处片段内的音素身份或可听范围，暂不确认音素。' : '我发现既有原唱/合成对应审计有误，暂不确认音素。';
    const answer = {observation_schema: OBSERVATION, review_id: item.review_id, status: draft.status, note_id: confirmed ? draft.note_id : null,
      phones: confirmed ? draft.phones.map(p => ({...p, edge_status: p.edge_status === 'truncated' ? 'partial' : p.edge_status})) : [], statement: prefix + (draft.statement.trim() ? '\n补充原话：' + draft.statement.trim() : ''),
      audible_identity_confirmed: confirmed ? draft.audible_identity_confirmed : false};
    return validateAnswer(packet, answer, reviewer);
  }
  function validateImport(packet, payload) {
    checkPacket(packet);
    const allowed = packet.schema_version === TRAIN_PACKET ? [TRAIN_ANSWER] : [SCHEMA, 'original_edge_review_answers_v1'];
    demand(payload && allowed.includes(payload.schema_version), '答案文件格式不匹配');
    demand(payload.packet_sha256 === packet.packet_sha256, '答案与当前 packet 哈希不匹配');
    demand(typeof payload.reviewer === 'string', 'reviewer 必须为字符串');
    demand(typeof payload.exported_at === 'string' && Number.isFinite(Date.parse(payload.exported_at)), '导出时间无效');
    demand(Array.isArray(payload.answers), 'answers 必须为列表');
    const ids = new Set();
    const answers = payload.answers.map(answer => {
      demand(!ids.has(answer.review_id), '答案 review_id 重复'); ids.add(answer.review_id);
      demand(payload.schema_version !== 'original_edge_review_answers_v1' || answer.observation_schema === undefined, '旧格式不能携带新版听觉记录');
      demand(payload.schema_version !== TRAIN_ANSWER || answer.observation_schema === OBSERVATION, '训练边缘须使用新版听觉记录');
      return validateAnswer(packet, answer, payload.reviewer);
    });
    return {...payload, answers};
  }
  function exportAnswers(packet, answers, reviewer, now = new Date().toISOString()) {
    return validateImport(packet, {schema_version: packet.schema_version === TRAIN_PACKET ? TRAIN_ANSWER : SCHEMA, packet_sha256: packet.packet_sha256, reviewer, exported_at: now, answers});
  }
  function playbackRange(item, preview = false) {
    if (preview) return {offset: 0, duration: item.duration_sec};
    return item.side === 'start' ? {offset: item.cut_in_preview_sec, duration: item.duration_sec - item.cut_in_preview_sec} :
      {offset: 0, duration: item.cut_in_preview_sec};
  }
  const api = {checkPacket, emptyDraft, selectNote, setPhone, validateAnswer, commitDraft, validateImport, exportAnswers, playbackRange};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.EdgeReviewState = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
