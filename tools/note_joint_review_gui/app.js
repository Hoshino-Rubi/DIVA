(function () {
  "use strict";
  const packet = JSON.parse(document.getElementById("packet").textContent);
  const packetSha = "__PACKET_SHA256__";
  const S = window.NoteReviewState;
  const $ = id => document.getElementById(id);
  const audio = $("player");
  let answer = S.fresh(packet, packetSha);
  let index = 0;
  let role = "mixed";
  let zoom = 1;
  let selected = null;
  let drag = null;
  let uncertainMode = false;
  let history = [];
  let future = [];
  const images = new Map();
  const WAVE_HEIGHT = 116;
  const PIANO_HEIGHT = 588;
  const MIDI_HIGH = 84;
  const MIDI_LOW = 36;

  function source() { return packet.clips[index]; }
  function clip() { return answer.clips[index]; }
  function notice(message, ok) {
    $("notice").textContent = message;
    $("notice").classList.toggle("ok", !!ok);
  }
  function save() {
    answer.reviewer = $("reviewer").value;
    answer.raw_statement = $("global-statement").value;
    try { localStorage.setItem(`note-review-${packetSha}`, JSON.stringify(answer)); }
    catch (_) { notice("浏览器本机存储不可用；请导出 JSON 保存进度"); }
  }
  function change(action) {
    const before = JSON.stringify(clip());
    try { action(); }
    catch (error) { answer.clips[index] = JSON.parse(before); loadControls(); throw error; }
    history.push(before);
    if (history.length > 80) history.shift();
    future = [];
    save();
    loadControls();
    refresh();
  }
  function rollback(stack, other) {
    if (!stack.length) return;
    other.push(JSON.stringify(clip()));
    answer.clips[index] = JSON.parse(stack.pop());
    selected = null;
    save();
    loadControls();
    refresh();
  }
  function relative(sample) { return S.relativeSec(source(), sample); }
  function absolute(seconds) { return S.absoluteSample(source(), seconds); }
  function limit(seconds) { return Math.max(0, Math.min(10, seconds)); }
  function fmt(seconds) { return seconds.toFixed(3); }
  function pitchY(midi) { return (MIDI_HIGH - midi) * 12; }
  function eventMidi(y) { return Math.min(MIDI_HIGH, Math.max(MIDI_LOW, MIDI_HIGH - Math.floor(y / 12))); }
  function eventSec(canvas, event) { return limit((event.clientX - canvas.getBoundingClientRect().left) / canvas.getBoundingClientRect().width * 10); }
  function x(sec, canvas) { return canvas.width * limit(sec) / 10; }
  function setDimensions() {
    const width = Math.max(880, Math.floor($("timeline-scroll").clientWidth - 2)) * zoom;
    for (const name of ["wave", "piano"]) {
      const canvas = $(name);
      canvas.width = width;
      canvas.style.width = `${width}px`;
    }
  }
  function paintWave() {
    const canvas = $("wave"), ctx = canvas.getContext("2d");
    const width = canvas.width;
    ctx.fillStyle = "#12282d";
    ctx.fillRect(0, 0, width, WAVE_HEIGHT);
    ctx.strokeStyle = "#2b5b60";
    ctx.beginPath();
    ctx.moveTo(0, WAVE_HEIGHT / 2);
    ctx.lineTo(width, WAVE_HEIGHT / 2);
    ctx.stroke();
    const peaks = source().waveforms[role];
    ctx.strokeStyle = role === "mixed" ? "#4cb2aa" : "#d2ae65";
    ctx.beginPath();
    peaks.forEach(([low, high], position) => {
      const column = position * width / peaks.length;
      ctx.moveTo(column, WAVE_HEIGHT / 2 - high * 47);
      ctx.lineTo(column, WAVE_HEIGHT / 2 - low * 47);
    });
    ctx.stroke();
    clip().uncertain_intervals.forEach(interval => {
      ctx.fillStyle = "#eb9c5066";
      ctx.fillRect(x(relative(interval.start_sample_abs), canvas), 0,
        x(relative(interval.end_sample_abs - interval.start_sample_abs + source().source_start_sample), canvas), WAVE_HEIGHT);
    });
    if (drag && drag.kind === "uncertain") {
      ctx.fillStyle = "#e99b5680";
      const left = Math.min(drag.start, drag.current), right = Math.max(drag.start, drag.current);
      ctx.fillRect(x(left, canvas), 0, x(right - left, canvas), WAVE_HEIGHT);
    }
    ctx.fillStyle = "#dce9e4";
    ctx.font = "12px system-ui";
    for (let second = 0; second <= 10; second++) ctx.fillText(`${second}s`, x(second, canvas) + 4, 16);
    ctx.strokeStyle = "#f16e59";
    ctx.beginPath(); ctx.moveTo(x(audio.currentTime || 0, canvas), 0);
    ctx.lineTo(x(audio.currentTime || 0, canvas), WAVE_HEIGHT); ctx.stroke();
  }
  function spectrum() {
    const key = `${index}-${role}`;
    if (!images.has(key)) {
      const img = new Image();
      img.onload = paint;
      img.src = source().preview[role].spectrogram_file;
      images.set(key, img);
    }
    return images.get(key);
  }
  function drawNote(ctx, canvas, note, ghost) {
    const left = x(relative(note.start_sample_abs), canvas);
    const right = x(relative(note.end_sample_abs), canvas);
    const top = pitchY(note.pitch_midi);
    ctx.fillStyle = ghost ? "#e9cb7988" : note.review_state === "confirmed" ? "#1f928bbb" :
      note.review_state === "uncertain" ? "#e99366bb" : "#e7c171bb";
    ctx.fillRect(left, top + 1, Math.max(2, right - left), 10);
    ctx.strokeStyle = note.id === selected ? "#fff" : "#142d32";
    ctx.lineWidth = note.id === selected ? 2 : 1;
    ctx.strokeRect(left, top + 1, Math.max(2, right - left), 10);
  }
  function paintPiano() {
    const canvas = $("piano"), ctx = canvas.getContext("2d");
    ctx.fillStyle = "#10262c";
    ctx.fillRect(0, 0, canvas.width, PIANO_HEIGHT);
    const image = spectrum();
    if (image.complete && image.naturalWidth) {
      ctx.globalAlpha = .76;
      ctx.drawImage(image, 0, 0, canvas.width, PIANO_HEIGHT);
      ctx.globalAlpha = 1;
    }
    for (let midi = MIDI_HIGH; midi >= MIDI_LOW; midi--) {
      const row = pitchY(midi);
      ctx.strokeStyle = midi % 12 === 0 ? "#c8e4d067" : "#92c1bc24";
      ctx.beginPath(); ctx.moveTo(0, row); ctx.lineTo(canvas.width, row); ctx.stroke();
      if (midi % 12 === 0) {
        ctx.fillStyle = "#e2e9e5";
        ctx.font = "12px system-ui";
        ctx.fillText(`${S.name(midi)} · ${midi}`, 5, row + 12);
      }
    }
    for (let second = 0; second <= 10; second++) {
      ctx.strokeStyle = "#b1d0cd44";
      ctx.beginPath(); ctx.moveTo(x(second, canvas), 0); ctx.lineTo(x(second, canvas), PIANO_HEIGHT); ctx.stroke();
    }
    clip().notes.forEach(note => drawNote(ctx, canvas, note, false));
    if (drag && drag.preview) drawNote(ctx, canvas, drag.preview, true);
    ctx.strokeStyle = "#f16e59";
    ctx.beginPath(); ctx.moveTo(x(audio.currentTime || 0, canvas), 0);
    ctx.lineTo(x(audio.currentTime || 0, canvas), PIANO_HEIGHT); ctx.stroke();
  }
  function paint() { paintWave(); paintPiano(); }

  function nav() {
    const host = $("songs");
    host.replaceChildren();
    packet.clips.forEach((item, number) => {
      const button = document.createElement("button");
      const state = answer.clips[number];
      button.classList.toggle("active", number === index);
      button.textContent = `${number + 1}. ${item.title}`;
      const small = document.createElement("small");
      small.textContent = `${state.review_state === "complete" ? "已核完" : state.review_state === "uncertain" ? "不确定" : "待定"} · ${state.notes.length} 音符`;
      button.append(small);
      button.addEventListener("click", () => { if (number !== index) openSong(number); });
      host.append(button);
    });
  }
  function loadControls() {
    const entry = clip();
    $("review-state").value = entry.review_state;
    $("all-notes-reviewed").checked = entry.all_notes_reviewed;
    $("silent-confirmed").checked = entry.silent_confirmed;
    $("clip-statement").value = entry.raw_statement;
  }
  function openSong(number) {
    audio.pause();
    index = number;
    selected = null;
    drag = null;
    history = []; future = [];
    role = "mixed";
    const item = source();
    $("song-title").textContent = item.title;
    $("source-info").textContent = `${item.example_id} · 原唱 ${item.source_start_sample}–${item.source_end_sample_exclusive} 样本 · ${item.source_sample_rate} Hz`;
    loadControls();
    changeAudio();
    nav();
    refresh();
  }
  function changeAudio() {
    const time = Number.isFinite(audio.currentTime) ? audio.currentTime : 0;
    audio.src = source().preview[role].file;
    audio.load();
    audio.currentTime = limit(time);
    $("source-mixed").classList.toggle("active", role === "mixed");
    $("source-vocal").classList.toggle("active", role === "vocal");
    paint();
  }
  function selectedNote() { return clip().notes.find(item => item.id === selected); }
  function renderUncertain() {
    const host = $("uncertain-list");
    host.replaceChildren();
    clip().uncertain_intervals.forEach((interval, number) => {
      const line = document.createElement("span");
      line.textContent = `${fmt(relative(interval.start_sample_abs))}–${fmt(relative(interval.end_sample_abs))} s: ${interval.reason}`;
      const remove = document.createElement("button");
      remove.textContent = "删除";
      remove.title = "删除这一不确定区";
      remove.addEventListener("click", () => change(() => clip().uncertain_intervals.splice(number, 1)));
      host.append(line, remove);
    });
  }
  function field(row, label, type, value, onChange) {
    const wrapper = document.createElement("label");
    wrapper.textContent = label;
    const input = document.createElement("input");
    input.type = type;
    input.value = value;
    if (type === "number") input.step = "1";
    input.addEventListener("change", () => {
      try { onChange(input.value); notice(""); }
      catch (error) { notice(error.message); renderNotes(); }
    });
    wrapper.append(input);
    row.append(wrapper);
    return wrapper;
  }
  function renderNotes() {
    const host = $("note-list");
    host.replaceChildren();
    const ordered = [...clip().notes].sort((a, b) => a.start_sample_abs - b.start_sample_abs || a.end_sample_abs - b.end_sample_abs);
    $("note-count").textContent = `${ordered.length} 项 · 已确认 ${ordered.filter(note => note.review_state === "confirmed").length} 项`;
    if (!ordered.length) {
      const empty = document.createElement("div");
      empty.className = "empty";
      empty.textContent = "暂无人工音符标注";
      host.append(empty);
    }
    ordered.forEach((note, number) => {
      const row = document.createElement("div");
      row.className = `note-row${note.id === selected ? " selected" : ""}`;
      row.addEventListener("click", () => {
        if (selected !== note.id) { selected = note.id; renderNotes(); paint(); }
      });
      const id = document.createElement("span"); id.className = "id"; id.textContent = String(number + 1);
      row.append(id);
      const midi = field(row, "MIDI", "number", note.pitch_midi, value => change(() => {
        const pitch = Number(value);
        if (!Number.isInteger(pitch) || pitch < 0 || pitch > 127) throw new Error("MIDI 必须是 0–127 的整数");
        note.pitch_midi = pitch; note.review_state = "draft"; S.record(clip(), "pitch", note.id);
      }));
      const name = document.createElement("small"); name.className = "pitch-name"; name.textContent = S.name(note.pitch_midi); midi.append(name);
      for (const [key, label] of [["start_sample_abs", "原唱起点样本"], ["end_sample_abs", "原唱终点样本"]]) {
        const cell = field(row, label, "number", note[key], value => change(() => {
          const sample = Number(value);
          const nextStart = key === "start_sample_abs" ? sample : note.start_sample_abs;
          const nextEnd = key === "end_sample_abs" ? sample : note.end_sample_abs;
          if (!Number.isInteger(sample) || nextStart < source().source_start_sample ||
              nextEnd > source().source_end_sample_exclusive || nextStart >= nextEnd) throw new Error("样本越界或时值非正");
          note[key] = sample; note.review_state = "draft"; S.record(clip(), "time", note.id);
        }));
        const seconds = document.createElement("small"); seconds.className = "sample";
        seconds.textContent = `${fmt(relative(note[key]))} s`; cell.append(seconds);
      }
      const edge = document.createElement("select"); edge.className = "edge"; edge.title = "切边可听状态";
      for (const [value, label] of [["fully_audible", "完整"], ["left_censored", "左截断"],
        ["right_censored", "右截断"], ["both_censored", "两侧截断"], ["uncertain", "边缘不确定"]]) {
        const option = document.createElement("option"); option.value = value; option.textContent = label;
        edge.append(option);
      }
      edge.value = note.edge_status;
      edge.addEventListener("change", () => {
        try {
          change(() => {
            if (edge.value.includes("left") && note.start_sample_abs !== source().source_start_sample ||
                edge.value.includes("right") && note.end_sample_abs !== source().source_end_sample_exclusive ||
                edge.value === "both_censored" && (note.start_sample_abs !== source().source_start_sample ||
                  note.end_sample_abs !== source().source_end_sample_exclusive)) throw new Error("截断标记必须紧邻窗口边缘");
            note.edge_status = edge.value; note.review_state = "draft";
          });
          notice("");
        } catch (error) { notice(error.message); }
      });
      row.append(edge);
      const status = document.createElement("select"); status.className = "confirm"; status.title = "逐音符确认状态";
      for (const [value, label] of [["draft", "未确认"], ["confirmed", "确认"], ["uncertain", "不确定"]]) {
        const option = document.createElement("option"); option.value = value; option.textContent = label; status.append(option);
      }
      status.value = note.review_state;
      status.addEventListener("change", () => change(() => { note.review_state = status.value; S.record(clip(), status.value, note.id); }));
      row.append(status);
      const statement = field(row, "原话", "text", note.raw_statement, value => change(() => { note.raw_statement = value; }));
      statement.className = "statement";
      const details = document.createElement("div"); details.className = "details";
      field(details, "重叠理由", "text", note.overlap_reason, value => change(() => { note.overlap_reason = value; }));
      row.append(details);
      host.append(row);
    });
  }
  function refresh() {
    nav();
    $("clip-state").textContent = clip().review_state === "complete" ? "人工已核完" :
      clip().review_state === "uncertain" ? "保留不确定" : "待确认";
    renderUncertain();
    renderNotes();
    paint();
  }

  function noteAt(seconds, midi) {
    return [...clip().notes].reverse().find(note => note.pitch_midi === midi &&
      seconds >= relative(note.start_sample_abs) && seconds <= relative(note.end_sample_abs));
  }
  function pianoDown(event) {
    const canvas = $("piano");
    const sec = eventSec(canvas, event);
    const midi = eventMidi(event.clientY - canvas.getBoundingClientRect().top);
    const hit = noteAt(sec, midi);
    if (hit) {
      selected = hit.id;
      const left = relative(hit.start_sample_abs), right = relative(hit.end_sample_abs);
      const edge = 9 / canvas.getBoundingClientRect().width * 10;
      const kind = Math.abs(sec - left) < edge ? "resize-left" :
        Math.abs(sec - right) < edge ? "resize-right" : "move";
      drag = {kind, start: sec, current: sec, midi, original: {...hit}, preview: null};
    } else {
      selected = null;
      drag = {kind: "draw", start: sec, current: sec, midi, preview: null};
    }
    canvas.setPointerCapture(event.pointerId);
    renderNotes(); paint();
  }
  function pianoMove(event) {
    const canvas = $("piano");
    const sec = eventSec(canvas, event);
    const midi = eventMidi(event.clientY - canvas.getBoundingClientRect().top);
    $("readout").textContent = `时间 ${fmt(sec)} s · MIDI ${midi} (${S.name(midi)})`;
    if (!drag) return;
    drag.current = sec;
    const base = drag.original;
    let left, right, pitch;
    if (drag.kind === "draw") {
      left = Math.min(drag.start, sec); right = Math.max(drag.start, sec);
      if (right - left < .025) right = limit(left + .1);
      pitch = drag.midi;
    } else {
      left = relative(base.start_sample_abs); right = relative(base.end_sample_abs); pitch = base.pitch_midi;
      if (drag.kind === "resize-left") left = Math.min(sec, right - .001);
      if (drag.kind === "resize-right") right = Math.max(sec, left + .001);
      if (drag.kind === "move") {
        const length = right - left;
        left = Math.max(0, Math.min(10 - length, left + sec - drag.start));
        right = left + length;
        pitch = Math.min(127, Math.max(0, base.pitch_midi + midi - drag.midi));
      }
    }
    drag.preview = {id: "preview", start_sample_abs: absolute(left), end_sample_abs: absolute(right), pitch_midi: pitch};
    paint();
  }
  function pianoUp(event) {
    if (!drag) return;
    pianoMove(event);
    const move = drag;
    drag = null;
    if (move.kind === "draw") {
      change(() => { selected = S.addNote(clip(), source(), relative(move.preview.start_sample_abs),
        relative(move.preview.end_sample_abs), move.preview.pitch_midi).id; });
    } else if (Math.abs(move.current - move.start) > .0005 || move.preview.pitch_midi !== move.original.pitch_midi) {
      change(() => {
        const target = selectedNote();
        target.start_sample_abs = move.preview.start_sample_abs;
        target.end_sample_abs = move.preview.end_sample_abs;
        target.pitch_midi = move.preview.pitch_midi;
        target.edge_status = "fully_audible";
        target.review_state = "draft";
        S.record(clip(), move.kind, target.id);
      });
    } else paint();
  }
  function waveDown(event) {
    const canvas = $("wave"), sec = eventSec(canvas, event);
    if (uncertainMode) {
      drag = {kind: "uncertain", start: sec, current: sec};
      canvas.setPointerCapture(event.pointerId);
    } else {
      audio.currentTime = sec; paint();
    }
  }
  function waveMove(event) {
    if (!drag || drag.kind !== "uncertain") return;
    drag.current = eventSec($("wave"), event); paint();
  }
  function waveUp(event) {
    if (!drag || drag.kind !== "uncertain") return;
    waveMove(event);
    const {start, current} = drag;
    drag = null;
    const reason = $("uncertain-reason").value.trim();
    if (Math.abs(start - current) < .025 || !reason) {
      notice("不确定区需要拖动至少 25 ms，并填写原因"); paint(); return;
    }
    change(() => clip().uncertain_intervals.push({start_sample_abs: absolute(Math.min(start, current)),
      end_sample_abs: absolute(Math.max(start, current)), reason}));
  }

  $("piano").addEventListener("pointerdown", pianoDown);
  $("piano").addEventListener("pointermove", pianoMove);
  $("piano").addEventListener("pointerup", pianoUp);
  $("piano").addEventListener("pointercancel", () => { drag = null; paint(); });
  $("wave").addEventListener("pointerdown", waveDown);
  $("wave").addEventListener("pointermove", waveMove);
  $("wave").addEventListener("pointerup", waveUp);
  $("wave").addEventListener("pointercancel", () => { drag = null; paint(); });
  $("play").addEventListener("click", async () => {
    if (!audio.paused) audio.pause();
    else {
      try { await audio.play(); notice(""); }
      catch (_) { notice("音频不能播放；请检查同目录 media 文件和浏览器权限"); }
    }
  });
  audio.addEventListener("play", () => { $("play").textContent = "暂停"; });
  audio.addEventListener("pause", () => { $("play").textContent = "播放"; });
  audio.addEventListener("timeupdate", () => {
    const note = selectedNote();
    if ($("loop").checked && note && audio.currentTime >= relative(note.end_sample_abs)) {
      audio.currentTime = relative(note.start_sample_abs);
    }
    $("playhead").value = String(Math.min(10, audio.currentTime));
    $("position").textContent = `${fmt(audio.currentTime)} / 10.000 s`;
    paint();
  });
  $("playhead").addEventListener("input", event => { audio.currentTime = Number(event.target.value); paint(); });
  $("source-mixed").addEventListener("click", () => { role = "mixed"; changeAudio(); });
  $("source-vocal").addEventListener("click", () => { role = "vocal"; changeAudio(); });
  $("zoom").addEventListener("input", event => { zoom = Number(event.target.value); setDimensions(); paint(); });
  $("undo").addEventListener("click", () => rollback(history, future));
  $("redo").addEventListener("click", () => rollback(future, history));
  $("uncertain-mode").addEventListener("click", () => {
    uncertainMode = !uncertainMode;
    $("uncertain-mode").classList.toggle("active", uncertainMode);
  });
  $("split").addEventListener("click", () => {
    const note = selectedNote(), at = absolute(audio.currentTime);
    if (!note || at <= note.start_sample_abs || at >= note.end_sample_abs) { notice("先选中音符，再把播放位置移到音符内部"); return; }
    change(() => {
      const end = note.end_sample_abs;
      note.end_sample_abs = at; note.review_state = "draft";
      const next = S.addNote(clip(), source(), relative(at), relative(end), note.pitch_midi);
      next.edge_status = note.edge_status === "right_censored" ? "right_censored" : "fully_audible";
      note.edge_status = note.edge_status === "right_censored" ? "fully_audible" : note.edge_status;
      S.record(clip(), "split", note.id);
      selected = next.id;
    });
  });
  $("merge").addEventListener("click", () => {
    const note = selectedNote();
    const ordered = [...clip().notes].sort((a, b) => a.start_sample_abs - b.start_sample_abs);
    const next = ordered[ordered.indexOf(note) + 1];
    if (!note || !next || note.pitch_midi !== next.pitch_midi || next.start_sample_abs < note.end_sample_abs) {
      notice("请选择可顺序合并的两个同音音符"); return;
    }
    change(() => {
      note.end_sample_abs = next.end_sample_abs;
      note.edge_status = next.edge_status === "right_censored" ? "right_censored" : note.edge_status;
      note.review_state = "draft";
      clip().notes.splice(clip().notes.indexOf(next), 1);
      S.record(clip(), "merge", note.id);
    });
  });
  $("delete").addEventListener("click", () => {
    const note = selectedNote();
    if (!note) { notice("请先选中一个音符"); return; }
    change(() => { clip().notes.splice(clip().notes.indexOf(note), 1); S.record(clip(), "delete", note.id); selected = null; });
  });
  $("review-state").addEventListener("change", event => change(() => { clip().review_state = event.target.value; }));
  $("all-notes-reviewed").addEventListener("change", event => change(() => { clip().all_notes_reviewed = event.target.checked; }));
  $("silent-confirmed").addEventListener("change", event => {
    if (event.target.checked && clip().notes.length) { event.target.checked = false; notice("整段静音不能同时保留音符"); return; }
    change(() => { clip().silent_confirmed = event.target.checked; });
  });
  $("clip-statement").addEventListener("change", event => change(() => { clip().raw_statement = event.target.value; }));
  $("reviewer").addEventListener("change", save);
  $("global-statement").addEventListener("change", save);
  $("save").addEventListener("click", () => { save(); notice("草稿已写入浏览器本机存储；请导出 JSON 备份", true); });
  $("export").addEventListener("click", () => {
    save();
    const blob = new Blob([JSON.stringify(answer, null, 2) + "\n"], {type: "application/json"});
    const url = URL.createObjectURL(blob), link = document.createElement("a");
    link.href = url; link.download = `original-note-answers-${packetSha.slice(0, 12)}.json`;
    link.click(); setTimeout(() => URL.revokeObjectURL(url), 3000);
    notice("五段答案已导出；待定与不确定项原样保留", true);
  });
  $("import").addEventListener("change", async event => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      answer = S.validateImport(packet, packetSha, JSON.parse(await file.text()));
      $("reviewer").value = answer.reviewer;
      $("global-statement").value = answer.raw_statement;
      save(); openSong(0);
      notice("五段答案已导入", true);
    } catch (error) { notice(`导入失败：${error.message}`); }
    event.target.value = "";
  });
  window.addEventListener("resize", () => { setDimensions(); paint(); });

  try {
    const saved = localStorage.getItem(`note-review-${packetSha}`);
    if (saved) answer = S.validateImport(packet, packetSha, JSON.parse(saved));
  } catch (_) { /* A corrupt local draft never overrides the frozen packet. */ }
  $("reviewer").value = answer.reviewer;
  $("global-statement").value = answer.raw_statement;
  setDimensions();
  openSong(0);
})();
