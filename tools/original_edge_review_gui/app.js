(function () {
  'use strict';
  const S = window.EdgeReviewState;
  const packet = S.checkPacket(JSON.parse(document.getElementById('packet').textContent));
  const audioData = JSON.parse(document.getElementById('audio-data').textContent);
  const storageKey = 'original-edge-review:' + packet.packet_sha256;
  const $ = id => document.getElementById(id);
  let current = 0, drafts = {}, answers = {}, source = null, context = null, playbackEpoch = 0;
  const buffers = new Map();
  function el(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  }
  function button(text, action, className) { const node = el('button', text, className); node.type = 'button'; node.addEventListener('click', action); return node; }
  function notice(text, error = false) { $('notice').textContent = text; $('notice').className = error ? 'error' : ''; }
  function draft() { return drafts[packet.items[current].review_id] ||= S.emptyDraft(); }
  function change(value) {
    const id = packet.items[current].review_id; drafts[id] = value; delete answers[id]; save(false);
  }
  function save(loud = true) {
    try {
      localStorage.setItem(storageKey, JSON.stringify({packet_sha256: packet.packet_sha256, current, drafts, answers, reviewer: $('reviewer').value}));
      if (loud) notice('本机草稿已保存；尚未提交的草稿不包含在答案 JSON 中。');
    } catch (_) { notice('浏览器不允许本机存储。请提交答案并导出 JSON，避免丢失。', true); }
  }
  function stop() {
    playbackEpoch += 1;
    if (source) { try { source.stop(); } catch (_) {} source = null; }
  }
  async function decoded(file) {
    if (!context) context = new (window.AudioContext || window.webkitAudioContext)();
    if (!buffers.has(file)) {
      if (typeof audioData[file] !== 'string') throw new Error('离线音频未嵌入：' + file);
      const bytes = Uint8Array.from(atob(audioData[file]), char => char.charCodeAt(0));
      buffers.set(file, context.decodeAudioData(bytes.buffer));
    }
    return buffers.get(file);
  }
  async function play(item, media, preview, canvas) {
    stop(); const epoch = playbackEpoch;
    try {
      const buffer = await decoded(media.audio_file);
      await context.resume();
      if (epoch !== playbackEpoch) return;
      const range = S.playbackRange(item, preview);
      if (buffer.duration + 0.001 < range.offset + range.duration) throw new Error('音频不足 2.5 秒，拒绝播放错误切点');
      drawWave(canvas, buffer, item);
      source = context.createBufferSource(); source.buffer = buffer; source.connect(context.destination);
      source.start(0, range.offset, range.duration);
      notice(preview ? '正在播放完整预览；预览外缘不是正式切点。' :
        '正在播放正式片段内侧：' + (item.side === 'start' ? '仅中线右侧（1.25–2.50 秒）' : '仅中线左侧（0–1.25 秒）'));
    } catch (error) { notice('播放失败：' + error.message, true); }
  }
  function drawWave(canvas, buffer, item) {
    const ctx = canvas.getContext('2d'), w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#d9efea'; ctx.fillRect(item.side === 'start' ? w / 2 : 0, 0, w / 2, h);
    if (buffer) {
      const data = buffer.getChannelData(0); ctx.strokeStyle = '#426970'; ctx.beginPath();
      for (let x = 0; x < w; x++) {
        let peak = 0; const a = Math.floor(x * data.length / w), b = Math.floor((x + 1) * data.length / w);
        for (let i = a; i < b; i++) peak = Math.max(peak, Math.abs(data[i]));
        ctx.moveTo(x, h / 2 - peak * h * .45); ctx.lineTo(x, h / 2 + peak * h * .45);
      }
      ctx.stroke();
    }
    ctx.strokeStyle = '#b72727'; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h); ctx.stroke();
  }
  function mediaPanel(item, media) {
    const original = media.role === 'original';
    const panel = el('article', undefined, 'panel ' + (original ? 'original' : ''));
    panel.append(el('h3', original ? '原唱 · 主听审音频' : media.role === 'rendered' ? '合成 · 仅辅助辨认' : '辅助音频'));
    const canvas = el('canvas'); canvas.width = 1000; canvas.height = 110;
    canvas.setAttribute('aria-label', '2.5 秒音频预览，红色中线为正式切点；绿色为片段内侧');
    panel.append(canvas); drawWave(canvas, null, item);
    panel.append(el('p', '中线 1.25 秒 = 正式切点（原唱时间 ' + item.original_cut_sec + ' 秒）', 'center-label'));
    const labels = el('div', undefined, 'cut-labels');
    labels.append(el('span', '0 秒 · 预览外缘 ≠ 正式切点'), el('span', '2.50 秒 · 预览外缘 ≠ 正式切点')); panel.append(labels);
    const controls = el('div', undefined, 'actions');
    controls.append(button('播放片段内侧 · ' + (item.side === 'start' ? '仅中线右侧' : '仅中线左侧'), () => play(item, media, false, canvas), original ? 'primary' : ''),
      button('完整 2.5 秒预览（不是正式片段）', () => play(item, media, true, canvas), 'preview-button'), button('停止', stop));
    panel.append(controls, el('p', '绿色区域才是本项需要判断的范围。播放按音频采样时长精确截在中线，不使用播放器时间轮询。', 'muted'));
    return panel;
  }
  function renderNavigation() {
    $('items').replaceChildren();
    packet.items.forEach((item, i) => {
      $('items').append(button((i + 1) + '/' + packet.items.length + ' · ' + item.review_id + (answers[item.review_id] ? ' ✓ 已提交' : ' · 待提交'),
        () => { stop(); save(false); current = i; render(); }, i === current ? 'active' : ''));
    });
  }
  function render() {
    const scrollX = window.scrollX || 0, scrollY = window.scrollY || 0;
    renderContents();
    if (window.scrollTo) window.scrollTo({left: scrollX, top: scrollY, behavior: 'instant'});
  }
  function renderContents() {
    renderNavigation();
    const item = packet.items[current], d = draft(), root = $('review'); root.replaceChildren();
    const intro = el('article');
    intro.append(el('h2', item.review_id + ' · ' + (item.side === 'start' ? '开头首字 / 首音素' : '结尾末字 / 末音素')),
      el('p', item.question || '中线处，正式片段内实际听到了哪些音素？'),
      el('p', item.side === 'start' ? '开头题：听中线右侧。按实际听感记录，不要求缺失音素必须位于某个索引。' :
        '结尾题：听中线左侧。按实际听感记录，不要求缺失音素必须位于某个索引。'),
      el('p', '内部沿用已校准的连续音符，不要求你重标全曲。这里不能凭文字自动确定可听音素。', 'muted'));
    const audit = el('details'); audit.open = true;
    audit.append(el('summary', '已完成的对应审计 · 先前原话（只读，不重新询问）'));
    const previous = item.previous_audit;
    if (typeof previous === 'string') audit.append(el('pre', previous));
    else {
      audit.append(el('p', '最终对应结论（既有记录）：' + (previous.decision || '未记录')));
      const prior = previous.previous_human_response || {}, final = previous.final_audit_response || {};
      audit.append(el('h3', '先前提交原话'), el('pre', prior.review_statement || prior.statement || '（无原话）'),
        el('h3', '最终审计原话'), el('pre', final.comment || final.review_statement || final.statement || '（无原话）'));
      for (const [key, label] of [['first_heard', '先前首字描述'], ['last_heard', '先前末字描述']]) {
        if (final[key]) audit.append(el('p', label), el('pre', final[key]));
      }
      audit.append(el('p', '原始完整审计对象保留在 packet.json；这里的原话不是新的音素确认。', 'muted'));
    }
    intro.append(audit); root.append(intro);
    [...item.media].sort((a, b) => (a.role !== 'original') - (b.role !== 'original')).forEach(m => root.append(mediaPanel(item, m)));
    const form = el('article'); form.append(el('h3', '1 · 手动选择中线处的一个边缘 note'));
    const notes = el('div', undefined, 'notes');
    item.notes.forEach(note => {
      const b = button('', () => { change(S.selectNote(draft(), note.note_id)); render(); }, 'note' + (d.note_id === note.note_id ? ' active' : ''));
      b.append(el('span', note.lyric + ' · /' + note.phones.join(' ') + '/'), el('small', 'note_id（只读）：' + note.note_id), el('small', '代理映射：' + note.mapping_status)); notes.append(b);
    });
    form.append(notes);
    const note = item.notes.find(n => n.note_id === d.note_id);
    form.append(el('h3', '2 · 逐个记录 IPA 的实际听感'));
    if (!note) form.append(el('p', '尚未选择 note；没有默认选中的音素。', 'muted'));
    else {
      form.append(el('p', '完整可听 = 整个音可听；部分可听 = 还剩一部分；完全听不到 = 整个音没有听到；不确定 = 无法判断。未判断不会被当作听不到，也不自动推断是切片造成的。', 'muted'));
      form.append(button('全部可听（显式选择全部为完整）', () => {
        change({...draft(), observation_schema: 'phone_audibility_v2', phones: note.phones.map((_, i) => ({phone_index: i, edge_status: 'full'})), audible_identity_confirmed: false}); render();
      }), button('清空音素选择', () => { change({...draft(), phones: [], audible_identity_confirmed: false}); render(); }));
      note.phones.forEach((phone, index) => {
        const row = el('div', undefined, 'phone-row'), selected = d.phones.find(p => p.phone_index === index);
        row.append(el('strong', '/' + phone + '/', 'ipa'), el('small', 'phone_index（只读）：' + index));
        const options = el('div', undefined, 'phone-options');
        const selectedStatus = selected?.edge_status === 'truncated' ? 'partial' : (selected?.edge_status || '');
        [['', '未判断'], ['full', '完整可听'], ['partial', '部分可听'], ['absent', '完全听不到'], ['uncertain', '不确定']].forEach(([value, label]) => {
          const choice = button(label, () => {
            change(S.setPhone(draft(), index, value));
            for (const sibling of options.children) sibling.className = sibling === choice ? 'active' : '';
            checkbox.checked = false;
            renderNavigation();
          }, selectedStatus === value ? 'active' : '');
          options.append(choice);
        }); row.append(options); form.append(row);
      });
    }
    const statusLabel = el('label', '3 · 本项结论（必须明确选择）', 'status-label');
    const status = el('select'); status.id = 'answer-status'; statusLabel.htmlFor = status.id;
    [['', '请选择结论'], ['confirmed', '确认并提交此字的听觉记录'], ['uncertain', '不确定，暂不确认音素'], ['wrong_correspondence', '例外：发现既有对应审计有误']].forEach(([value, label]) => {
      const option = el('option', label); option.value = value; status.append(option);
    }); status.value = d.status;
    status.addEventListener('change', () => {
      const value = status.value;
      change(value === 'confirmed' ? {...draft(), status: value, audible_identity_confirmed: false} :
        {...draft(), status: value, note_id: null, phones: [], audible_identity_confirmed: false}); render();
    });
    form.append(statusLabel, status);
    const affirmation = el('label', undefined, 'affirmation'), checkbox = el('input'); checkbox.type = 'checkbox';
    checkbox.checked = d.audible_identity_confirmed; checkbox.disabled = d.status !== 'confirmed';
    checkbox.addEventListener('change', () => { change({...draft(), audible_identity_confirmed: checkbox.checked}); renderNavigation(); });
    affirmation.append(checkbox, document.createTextNode(' 我明确确认：所选字是中线处片段内的' + (item.side === 'start' ? '首字' : '末字') + '；完整可听、部分可听、完全听不到和不确定均按实际听感填写，不按歌词推测，不自动认定缺失原因。'));
    const statement = el('textarea', undefined, 'statement'); statement.value = d.statement; statement.placeholder = '可选：补充你的原话或不确定的原因（原样保留）';
    statement.setAttribute('aria-label', '补充原话');
    statement.addEventListener('input', () => { change({...draft(), statement: statement.value}); renderNavigation(); });
    form.append(affirmation, statement);
    const controls = el('div', undefined, 'actions');
    controls.append(button('提交本项并下一项', () => {
      try {
        answers[item.review_id] = S.commitDraft(packet, item, draft(), $('reviewer').value.trim());
        stop(); if (current < packet.items.length - 1) current += 1; save(false); render();
        notice('本项已提交。已提交 ' + Object.keys(answers).length + '/' + packet.items.length + ' 项；请导出 JSON 备份。');
      } catch (error) { notice(error.message, true); }
    }, 'primary'), button('仅保存草稿', () => save()), button('下一项（不提交）', () => {
      stop(); save(false); current = (current + 1) % packet.items.length; render(); notice('已切换；未提交的草稿不会导出。');
    })); form.append(controls); root.append(form);
    const binding = el('details'); binding.append(el('summary', '当前问卷绑定信息（只读）'), el('p', packet.packet_sha256, 'hash'),
      el('pre', JSON.stringify({example_id: item.example_id, review_id: item.review_id}, null, 2))); root.append(binding);
  }
  $('save').addEventListener('click', () => save());
  $('reviewer').addEventListener('input', () => save(false));
  $('export').addEventListener('click', () => {
    try {
      const payload = S.exportAnswers(packet, Object.values(answers), $('reviewer').value.trim());
      if (!payload.answers.length) throw new Error('尚无已提交答案，请先提交本项。');
      const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], {type: 'application/json'}));
      const a = el('a'); a.href = url; a.download = (packet.schema_version === 'replacement_edge_review_packet_v1' ? 'replacement-edge-review-answers-' : 'original-edge-review-answers-') + packet.packet_sha256.slice(0, 12) + '.json';
      document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 30000);
      notice('已导出 ' + payload.answers.length + ' 项已提交答案；未运行训练或自动晋升参考。');
    } catch (error) { notice(error.message, true); }
  });
  $('import').addEventListener('change', async event => {
    try {
      const file = event.target.files[0]; if (!file) return;
      const payload = S.validateImport(packet, JSON.parse(await file.text()));
      if (!window.confirm('导入 ' + payload.answers.length + ' 项答案？同 review_id 的草稿和已提交答案将被替换，其他项目保持不变。')) return;
      stop(); $('reviewer').value = payload.reviewer;
      payload.answers.forEach(answer => { answers[answer.review_id] = answer; drafts[answer.review_id] = structuredClone(answer); });
      save(false); render(); notice('导入成功，packet 绑定和所有答案均通过校验。');
    } catch (error) { notice('导入被拒绝：' + error.message, true); }
    finally { event.target.value = ''; }
  });
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey) || 'null');
    if (saved && saved.packet_sha256 === packet.packet_sha256) {
      const valid = S.exportAnswers(packet, Object.values(saved.answers || {}), saved.reviewer || '');
      $('reviewer').value = valid.reviewer; answers = Object.fromEntries(valid.answers.map(a => [a.review_id, a]));
      packet.items.forEach(item => {
        const d = saved.drafts?.[item.review_id];
        if (d && typeof d.statement === 'string' && Array.isArray(d.phones) &&
            (d.note_id === null || item.notes.some(n => n.note_id === d.note_id))) drafts[item.review_id] = d;
      });
      current = Number.isInteger(saved.current) && saved.current >= 0 && saved.current < packet.items.length ? saved.current : 0;
      notice('已恢复本机进度。旧“截断”仍表示部分可听；若你原意是整个音没有，请改选“完全听不到”并重新提交该项。');
    }
  } catch (_) { notice('本机进度不可用或未通过校验；未载入，请使用答案 JSON 恢复。', true); }
  render();
})();
