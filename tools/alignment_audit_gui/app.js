(() => {
  "use strict";
  const data = JSON.parse(document.getElementById("audit-data").textContent);
  const store = AlignmentAudit.createStore(data);
  const $ = id => document.getElementById(id);
  const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const sec = value => Number.isFinite(value) ? value.toFixed(3) + " s" : "未记录";
  const badge = (label, color = "") => '<span class="badge ' + color + '">' + esc(label) + '</span>';
  const humanLabel = {rejected:"人工已判错位", human_confirmed:"文字标为通过", pending:"人工待定"};
  const machineLabel = {machine_aligned:"旧机器标为通过", needs_review:"旧机器异常", machine_suggested:"机器建议 · 未确认"};
  const decisionLabel = {corresponds:"对应正确", misaligned:"错位", boundary_issue:"需修正边界", uncertain:"不确定"};
  const roleLabel = {original:"原唱 · 判定依据", rendered:"合成对应 · 仅作辅助", separated_vocal_auxiliary:"分离人声 · 仅作辅助"};
  const storageKey = "any2xsampa:alignment-audit:v1:" + data.bundle_id;
  let selectedId = null, activeRole = "original", draftDecision = "", dirty = false, visible = [], toastTimer;
  let storageWorks = true;
  function toast(message, error = false) {
    $("toast").textContent = message; $("toast").classList.toggle("error", error); $("toast").hidden = false;
    clearTimeout(toastTimer); toastTimer = setTimeout(() => { $("toast").hidden = true; }, 6000);
  }
  function persist() {
    try { localStorage.setItem(storageKey, JSON.stringify(store.exportData())); storageWorks = true; }
    catch (_) { storageWorks = false; }
    $("storage-status").textContent = storageWorks ? "已存本浏览器 · " + store.count() + " 项 · 请导出备份" : "浏览器存储不可用 · 请及时导出 JSON";
  }
  try {
    const saved = localStorage.getItem(storageKey);
    if (saved) store.importData(JSON.parse(saved));
  } catch (_) { storageWorks = false; toast("本浏览器缓存不可用或不匹配；未覆盖来源，请导入之前的审计备份", true); }
  $("storage-status").textContent = storageWorks ? "仅保存已点击“保存本项”的记录" : "请用导出按钮备份审计记录";
  const s = data.summary;
  $("summary").innerHTML = [
    [s.items,"审计片段","20 个训练候选 + 6 个沙龙切点",""],
    [s.human_rejected,"人工已判错位","默认隐藏，不要求重复审核","alert"],
    [s.human_machine_conflicts,"人机结论冲突","人工错位优先，机器结论不能覆盖","alert"],
    [s.training_text_approvals,"训练文字通过","仍非正式音素真值",""],
  ].map(([n,title,sub,color]) => '<div class="metric '+color+'"><b>'+n+'</b><span class="metric-label">'+title+'</span><small>'+sub+'</small></div>').join("");
  $("global-notice").textContent = "注意：旧报告称 11 项自动通过，但其中 " + s.human_machine_conflicts + " 项与你的错位结论冲突。这里不将任何旧机器结果自动计为可训练数据；旧报告未被改写。";
  $("dataset-date").textContent = "本地构建 " + new Date(data.built_at).toLocaleString("zh-CN", {hour12:false});
  const songs = [...new Map(data.rows.map(r => [r.song_id, r.song_name])).entries()];
  $("song").innerHTML += songs.map(([id,name])=>'<option value="'+esc(id)+'">'+esc(name)+'</option>').join("");
  function updateList() {
    visible = data.rows.filter(r => store.matches(r, $("filter").value, $("song").value, $("search").value));
    $("list-count").textContent = visible.length + " 项";
    $("item-list").innerHTML = visible.map(row => {
      const saved = store.get(row.review_id);
      const tag = saved ? badge("已保存", "green") : row.state.human_rejected ? badge("已判错位", "red") : row.kind === "boundary" ? badge("切点", "purple") : badge("待审计", "amber");
      const snippet = row.human.review_statement || "尚无人工文字意见";
      return '<button class="item-button '+(row.review_id===selectedId?'active':'')+'" data-select="'+esc(row.review_id)+'" aria-current="'+(row.review_id===selectedId?'true':'false')+'"><div class="row-top"><strong>'+esc(row.review_id)+'</strong>'+tag+'</div><div class="song-title">'+esc(row.song_name)+(row.state.human_machine_conflict?' · 结论冲突':'')+'</div><div class="snippet">'+esc(snippet)+'</div></button>';
    }).join("") || '<div class="empty">此筛选下没有片段。<br>可切换“全部”或“本次已保存”。</div>';
    $("item-list").querySelectorAll("[data-select]").forEach(b => b.addEventListener("click", () => select(b.dataset.select)));
  }
  function stopAll() { $("detail").querySelectorAll("audio").forEach(a=>a.pause()); }
  function select(id) {
    if (id !== selectedId && dirty && !window.confirm("本项有未保存内容。放弃这些改动并切换片段？")) return;
    stopAll(); dirty = false; selectedId = id; activeRole = "original"; updateList(); renderDetail();
  }
  function step(delta) {
    const index = visible.findIndex(r=>r.review_id === selectedId);
    const next = visible[index < 0 ? 0 : index + delta];
    if (next) select(next.review_id);
  }
  function playerHTML(row, media) {
    const cut = row.cut_in_preview_sec;
    const marker = Number.isFinite(cut) && media.duration_sec > 0 ? 100*(42/1200 + 1138/1200*(cut/media.duration_sec)) : null;
    return '<div class="player '+(media.role===activeRole?'active-player':'')+'" data-player="'+esc(media.role)+'"><div class="player-title"><span>'+esc(roleLabel[media.role] || media.role)+'</span><span class="muted">'+sec(media.duration_sec)+'</span></div><div class="wave" data-wave="'+esc(media.role)+'" title="点击波形跳转播放位置"><img src="'+esc(media.waveform_file)+'" alt="'+esc(row.review_id+' '+roleLabel[media.role]+' 波形')+'"><span class="playhead"></span>'+(marker!==null?'<span class="cutline" style="left:'+marker+'%"></span>':'')+'</div>'+(marker!==null?'<p class="cutlabel">虚线：原切点（预览 '+sec(cut)+'）· '+(row.side==='start'?'保留中线后侧作为片段开头':'保留中线前侧作为片段结尾')+'</p>':'')+'<audio controls preload="none" data-audio="'+esc(media.role)+'" src="'+esc(media.audio_file)+'"></audio><p class="player-error" hidden>音频无法加载，请保持 index.html 与 media 文件夹在一起。</p></div>';
  }
  function referenceHTML(row) {
    const near = new Set(row.nearby_note_ids);
    const refs = row.candidate_references.map(ref=>'<p>候选因子 '+esc(ref.factor)+'：合成 '+sec(ref.rendered_start_sec)+' – '+sec(ref.rendered_end_sec)+'，'+ref.note_ids.length+' 个音符（并非原唱音素时间）</p>').join("");
    const table = row.notes.map(note=>'<tr class="'+(near.has(note.note_id)?'nearby':'')+'"><td class="lyric">'+esc(note.lyric)+'</td><td class="noteid">'+esc(note.note_id)+'</td><td>'+esc(note.engine_tokens.join(" "))+'</td><td>'+note.phones.map((p,i)=>'<span class="chip"><sup>'+i+'</sup>'+esc(p)+'</span>').join("")+(note.mapping_status!=="resolved_proxy"?badge("映射未解决","red"):"")+'</td></tr>').join("");
    return '<details class="reference panel"><summary>汉字 ↔ 音符 ↔ IPA 对照 <span class="muted">'+row.notes.length+' 个源音符 · '+row.mapping_resolved_notes+' 个映射可解析 · 可选查看</span></summary><p class="muted">这是人工 VSQX 的派生映射，不是模型预测，也不证明边缘音素实际可听见。下标是该音符内从 0 开始的索引；本页不要求填写它。'+(row.kind==='boundary'?'浅色行是切点附近参考。':'')+'</p><div class="muted">'+refs+'</div><div class="table-wrap"><table><thead><tr><th>字</th><th>源 note_id</th><th>引擎标记</th><th>IPA / phone_index</th></tr></thead><tbody>'+table+'</tbody></table></div></details>';
  }
  function renderDetail() {
    stopAll();
    const row = data.rows.find(r=>r.review_id===selectedId);
    if (!row) { $("detail").innerHTML='<div class="panel empty"><h3>选择一个片段开始审计</h3><p>原始判断保留，不必重复录入。</p></div>'; return; }
    const saved = store.get(row.review_id);
    draftDecision = saved?.decision || "";
    const c = row.machine.confidence || {};
    const machineText = machineLabel[row.machine.status] || "无机器结果";
    let reason = row.machine.reason || "";
    if (reason.includes("correspondence resolutions disagree")) reason = "两套对应候选的首尾音符范围不一致，不能直接认定对应正确。";
    else if (reason.includes("unresolved note:")) {
      const note = row.notes.find(n => reason.includes(n.note_id));
      reason = "旧报告中有未解决的音素映射" + (note ? "：「" + note.lyric + "」（" + note.note_id + "）" : "。") ;
    }
    const time = row.kind === "boundary" ? '原唱切点 '+sec(row.original_cut_sec)+' · '+(row.side==='start'?'片段起点':'片段终点') : '原唱固定窗口 '+sec(row.start_sec)+' – '+sec(row.end_sec);
    const warning = row.warnings.map(w=>'<p class="inline-warning '+(row.state.human_machine_conflict?'conflict':'')+'">'+esc(w)+'</p>').join("");
    const confidenceText = Number.isFinite(c.score) ? '历史启发评分 '+c.score.toFixed(3)+'（不是准确率或正确概率）' : '未提供可解释的概率评分';
    const rendered = row.media.filter(m=>m.role!=="separated_vocal_auxiliary");
    const vocal = row.media.filter(m=>m.role==="separated_vocal_auxiliary");
    $("detail").innerHTML = '<section class="panel"><div class="detail-heading"><div class="id-block"><span class="id-token">'+esc(row.review_id)+'</span><h2>'+esc(row.song_name)+'</h2>'+badge(row.kind==='boundary'?'开发切点':'训练候选',row.kind==='boundary'?'purple':'')+'</div><div class="prevnext"><button id="prev" title="上一项 (J)">← 上一项</button><button id="next" title="下一项 (K)">下一项 →</button></div></div><p class="subtitle">'+time+'</p><div class="evidence-grid"><div class="evidence"><small>此前人工意见 · 原样保留</small>'+badge(humanLabel[row.human.status]||row.human.status,row.state.human_rejected?'red':'amber')+'<blockquote>'+esc(row.human.review_statement || '未填写说明')+'</blockquote></div><div class="evidence"><small>机器历史结果 · 仅作证据</small>'+badge(machineText,row.machine.status==='needs_review'?'amber':'')+'<p class="muted">'+esc(confidenceText)+'</p><p class="muted">端点分歧 '+sec(c.endpoint_disagreement_sec ?? c.resolution_disagreement_sec)+'</p>'+(row.machine.reason?'<p class="muted">'+esc(reason)+'</p>':'')+'</div></div>'+warning+'</section>'+
      '<section class="panel"><div class="section-heading"><h3>试听与对照</h3><span class="muted">音频保持历史版本 · 不自动移位</span></div><div class="audio-grid">'+rendered.map(m=>playerHTML(row,m)).join("")+'</div>'+(vocal.length?'<details class="reference"><summary>展开分离人声辅助</summary>'+vocal.map(m=>playerHTML(row,m)).join("")+'</details>':'')+'<div class="transport"><button id="play" class="primary">播放 / 暂停</button><button id="seek-back">−2 秒</button><button id="seek-forward">+2 秒</button><button id="switch-original">A 原唱</button><button id="switch-rendered">B 合成辅助</button><label for="speed">速度</label><select id="speed"><option value="0.75">0.75×</option><option value="1" selected>1.00×</option><option value="1.25">1.25×</option></select></div><div class="loop-settings"><label><input id="loop" type="checkbox">循环选区</label><label>从 <input id="loop-start" type="number" min="0" step="0.05" value="0">秒</label><label>到 <input id="loop-end" type="number" min="0.05" step="0.05" value="'+(row.media[0]?.duration_sec||10)+'">秒</label>'+(row.kind==='boundary'?'<button id="cut-before">只听中线前</button><button id="cut-after">只听中线后</button>':'')+'</div><p class="muted">A/B 按相对时间切换，不代表已对齐。切点选择、语速和循环仅改变试听，不改文件。</p></section>'+
      '<section class="panel"><div class="section-heading"><h3>本次审计判断</h3>'+badge(saved?'本项已有新记录':'尚未保存新判断',saved?'green':'')+'</div><div class="decision-grid">'+Object.entries(decisionLabel).map(([id,title])=>'<button data-decision="'+id+'" class="'+(id===draftDecision?'chosen':'')+'" aria-pressed="'+(id===draftDecision)+'">'+title+'</button>').join("")+'</div><div class="form-grid"><label>复核人<input id="reviewer" maxlength="120" value="'+esc(saved?.reviewer||row.human.reviewer||'')+'" placeholder="你的名字 / 昵称"></label><label>边界状态<select id="boundary"><option value="unknown">未判断边缘</option><option value="full">完整，没有截断</option><option value="truncated_start">开头被截断</option><option value="truncated_end">结尾被截断</option><option value="truncated_both">两端都被截断</option></select></label><label>开头听到什么（可不填）<input id="first-heard" maxlength="160" value="'+esc(saved?.first_heard||'')+'" placeholder="写字或短语，不填编号"></label><label>结尾听到什么（可不填）<input id="last-heard" maxlength="160" value="'+esc(saved?.last_heard||'')+'" placeholder="只填写能确定的部分"></label><label class="wide">补充说明（可不填）<textarea id="comment" maxlength="4000" placeholder="例如：只有后半段对应 / 第一个字丢失 / 整段错位">'+esc(saved?.comment||'')+'</textarea></label></div>'+(row.state.human_rejected?'<label class="revision-ack"><input id="revise" type="checkbox" '+(saved?.acknowledge_rejection_override?'checked':'')+'> 若改判为对应正确，我明确修订此项此前的错位结论（旧记录仍保留）</label>':'')+'<div class="form-actions"><span id="form-status" class="form-status">'+(saved?'保存时间 '+esc(new Date(saved.updated_at).toLocaleString()):'选择判断后点击保存；不会触发训练')+'</span><button id="reset-item">撤回本次判断</button><button id="save-item" class="primary">保存本项</button><button id="save-next">保存并下一项 →</button></div><p class="muted">“对应正确”仅确认对应关系；不会将整段所有音素自动确认为可听见。导出格式与正式音素真值格式分离。</p></section>'+
      referenceHTML(row)+'<details class="source-details panel"><summary>来源与审计证据（只读）</summary><dl class="data-grid"><dt>片段来源</dt><dd>'+esc(row.source_example_id)+'</dd><dt>复核包 SHA-256</dt><dd>'+esc(row.packet_sha256)+'</dd><dt>原唱 SHA-256</dt><dd>'+esc(row.source_audio_sha256)+'</dd><dt>VSQX SHA-256</dt><dd>'+esc(row.source_vsqx_sha256)+'</dd><dt>本页数据 ID</dt><dd>'+esc(data.bundle_id)+'</dd></dl><p class="muted">派生预览文件在构建时已逐个校验；未在页面中打开或修改外部原始整曲。</p><details><summary>查看原始人工记录</summary><pre class="raw">'+esc(JSON.stringify(row.human,null,2))+'</pre></details><details><summary>查看机器原始记录</summary><pre class="raw">'+esc(JSON.stringify(row.machine,null,2))+'</pre></details><details><summary>查看来源文件和哈希</summary><pre class="raw">'+esc(JSON.stringify(data.sources,null,2))+'</pre></details></details>';
    $("boundary").value = saved?.boundary_state || "unknown";
    $("prev").onclick = ()=>step(-1); $("next").onclick = ()=>step(1);
    $("detail").querySelectorAll("[data-decision]").forEach(b=>b.onclick=()=>{
      draftDecision=b.dataset.decision; dirty=true;
      $("detail").querySelectorAll("[data-decision]").forEach(x=>{x.classList.toggle("chosen",x===b);x.setAttribute("aria-pressed",String(x===b));});
      $("form-status").textContent="尚未保存";
    });
    ["reviewer","boundary","first-heard","last-heard","comment","revise"].forEach(id=>{
      if ($(id)) $(id).oninput=()=>{dirty=true;$("form-status").textContent="尚未保存";};
    });
    $("save-item").onclick=()=>saveCurrent(false); $("save-next").onclick=()=>saveCurrent(true);
    $("reset-item").onclick=()=>{
      if (!store.get(selectedId) && !dirty) return;
      if (!window.confirm("撤回本项本次填写的判断？历史人工意见不会改变。")) return;
      store.clear(selectedId);dirty=false;persist();updateList();renderDetail();
    };
    wireAudio(row);
  }
  function saveCurrent(goNext) {
    const idx=visible.findIndex(r=>r.review_id===selectedId), next=visible[idx+1]?.review_id;
    try {
      if (!draftDecision) throw Error("请先选择一个审计判断");
      store.save(selectedId,{decision:draftDecision,boundary_state:$("boundary").value,reviewer:$("reviewer").value,
        first_heard:$("first-heard").value,last_heard:$("last-heard").value,comment:$("comment").value,
        acknowledge_rejection_override:$("revise")?.checked||false});
      dirty=false;persist();toast(selectedId+" 已保存；原文件未修改"+(storageWorks?"":"。请导出备份"));
      if (goNext && next) select(next); else {updateList();renderDetail();}
    } catch (err) { toast(err.message,true); }
  }
  function audioFor(role=activeRole) { return $("detail").querySelector('[data-audio="'+role+'"]'); }
  async function switchAudio(role, forcePlay=false) {
    const previous=audioFor(), next=audioFor(role);
    if (!next) {toast("此片段没有该辅助音频");return;}
    const time=previous?.currentTime||0, shouldPlay=forcePlay || (previous&&!previous.paused);
    stopAll();activeRole=role; next.playbackRate=Number($("speed").value);
    $("detail").querySelectorAll("[data-player]").forEach(p=>p.classList.toggle("active-player",p.dataset.player===role));
    try { next.currentTime=Math.min(time,Number.isFinite(next.duration)?Math.max(0,next.duration-0.01):time); if(shouldPlay)await next.play(); }
    catch (_) {toast("音频暂时无法播放，请检查 media 文件夹",true);}
  }
  function loopRange(audio) {
    const row=data.rows.find(r=>r.review_id===selectedId), fallback=row.media.find(m=>m.role===activeRole)?.duration_sec||10;
    const max=Number.isFinite(audio.duration)?audio.duration:fallback;
    const a=Number($("loop-start").value),b=Number($("loop-end").value);
    return Number.isFinite(a)&&Number.isFinite(b)&&a>=0&&b>a&&b<=max+0.01 ? [a,Math.min(b,max)] : null;
  }
  async function togglePlay() {
    const a=audioFor();if(!a)return;
    if(!a.paused){a.pause();return;}
    const range=loopRange(a);
    if($("loop").checked&&!range){toast("循环选区必须在音频范围内，且结束大于开始",true);return;}
    if($("loop").checked&&(a.currentTime<range[0]||a.currentTime>=range[1]))a.currentTime=range[0];
    try{await a.play();}catch(_){toast("无法播放音频，请检查本地媒体文件",true);}
  }
  function wireAudio(row) {
    $("detail").querySelectorAll("[data-audio]").forEach(a=>{
      a.onplay=()=>{ $("detail").querySelectorAll("audio").forEach(x=>{if(x!==a)x.pause();});activeRole=a.dataset.audio;
        $("detail").querySelectorAll("[data-player]").forEach(p=>p.classList.toggle("active-player",p.dataset.player===activeRole));a.playbackRate=Number($("speed").value);};
      a.ontimeupdate=()=>{
        const duration=Number.isFinite(a.duration)?a.duration:row.media.find(m=>m.role===a.dataset.audio).duration_sec;
        a.closest(".player").querySelector(".playhead").style.left=(100*(42/1200+1138/1200*a.currentTime/duration))+"%";
        const range=loopRange(a);
        if(!a.paused&&$("loop").checked&&range&&a.currentTime>=range[1])a.currentTime=range[0];
      };
      a.onended=()=>{const r=loopRange(a);if($("loop").checked&&r){a.currentTime=r[0];a.play().catch(()=>{});}};
      a.onerror=()=>{a.closest(".player").querySelector(".player-error").hidden=false;};
    });
    $("detail").querySelectorAll("[data-wave]").forEach(w=>w.onclick=async event=>{
      await switchAudio(w.dataset.wave);const a=audioFor();const bounds=w.getBoundingClientRect();
      const fraction=Math.max(0,Math.min(1,((event.clientX-bounds.left)/bounds.width-42/1200)/(1138/1200)));
      const duration=Number.isFinite(a.duration)?a.duration:row.media.find(m=>m.role===activeRole).duration_sec;
      try{a.currentTime=fraction*duration;}catch(_){}
    });
    $("play").onclick=togglePlay;$("switch-original").onclick=()=>switchAudio("original");$("switch-rendered").onclick=()=>switchAudio("rendered");
    $("speed").onchange=()=>{ $("detail").querySelectorAll("audio").forEach(a=>a.playbackRate=Number($("speed").value));};
    for(const [id,delta] of [["seek-back",-2],["seek-forward",2]])$(id).onclick=()=>{const a=audioFor();if(a)a.currentTime=Math.max(0,Math.min(Number.isFinite(a.duration)?a.duration:Infinity,a.currentTime+delta));};
    if(row.kind==="boundary")for(const [id,after] of [["cut-before",false],["cut-after",true]])$(id).onclick=()=>{
      $("loop-start").value=after?row.cut_in_preview_sec:0;$("loop-end").value=after?row.media[0].duration_sec:row.cut_in_preview_sec;$("loop").checked=true;
      audioFor().currentTime=Number($("loop-start").value);if(audioFor().paused)togglePlay();
    };
  }
  for (const id of ["filter","song","search"]) $(id).addEventListener(id==="search"?"input":"change",()=>{ updateList();if(!selectedId&&visible[0])select(visible[0].review_id); });
  $("help-button").onclick=()=>{$("help").open=!$("help").open;};
  $("import-button").onclick=()=>$("import-file").click();
  $("import-file").onchange=async event=>{
    const file=event.target.files[0];if(!file)return;
    try{
      if(dirty&&!window.confirm("导入会刷新当前表单。放弃未保存的填写？"))return;
      if(file.size>2*1024*1024)throw Error("文件过大，审计 JSON 不应超过 2 MiB");
      const count=store.importData(JSON.parse(await file.text()));dirty=false;persist();updateList();renderDetail();toast("已导入 "+count+" 项审计判断");
    }catch(err){toast("导入失败："+err.message,true);}finally{event.target.value="";}
  };
  $("export-button").onclick=()=>{
    if(dirty){toast("请先保存当前片段，避免遗漏刚填写的意见",true);return;}
    const blob=new Blob([JSON.stringify(store.exportData(),null,2)+"\n"],{type:"application/json;charset=utf-8"});
    const a=document.createElement("a"),url=URL.createObjectURL(blob);a.href=url;
    a.download="alignment-audit-"+new Date().toISOString().replace(/[:.]/g,"-")+".json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    toast("已导出 "+store.count()+" 项新判断；不是音素真值文件");
  };
  window.addEventListener("beforeunload",event=>{if(dirty){event.preventDefault();event.returnValue="";}});
  document.addEventListener("keydown",event=>{
    if(event.ctrlKey||event.altKey||event.metaKey||/INPUT|TEXTAREA|SELECT|BUTTON/.test(event.target.tagName)||event.target.isContentEditable)return;
    const k=event.key.toLowerCase();if([" ","a","b","j","k"].includes(k))event.preventDefault();
    if(k===" ")togglePlay();if(k==="a")switchAudio("original");if(k==="b")switchAudio("rendered");if(k==="j")step(-1);if(k==="k")step(1);
  });
  updateList();if(visible[0])select(visible[0].review_id);else renderDetail();
})();
