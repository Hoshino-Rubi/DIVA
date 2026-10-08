/* Pure export validation is also runnable in Node; controls never rebuild the page. */
(function () {
  "use strict";
  function stable(value) {
    if (Array.isArray(value)) return value.map(stable);
    if (value && typeof value === "object") return Object.fromEntries(Object.keys(value).sort().map(k => [k, stable(value[k])]));
    return value;
  }
  function makeAnswers(wrapper, drafts, reviewer) {
    return {schema_version: "whole_sequence_review_answers_v1", packet_sha256: wrapper.packet_sha256,
      reviewer: String(reviewer || "user"), exported_at: new Date().toISOString(),
      answers: wrapper.packet.items.map(item => {
        const draft = drafts[item.review_id] || {};
        const requested = ["sequence_confirmed", "sequence_differs", "pending"].includes(draft.decision) ? draft.decision : "pending";
        const affirmed = draft.affirmed === true;
        return {review_id: item.review_id, status: requested === "sequence_confirmed" && !affirmed ? "pending" : requested,
          requested_decision: requested, explicit_sequence_scope_confirmed: affirmed,
          phone_timing_confirmed: false, statement: String(draft.statement || ""), source_bindings: item.source_bindings};
      })};
  }
  function parseAnswers(wrapper, payload) {
    if (!payload || payload.schema_version !== "whole_sequence_review_answers_v1" || payload.packet_sha256 !== wrapper.packet_sha256)
      throw new Error("回答不是当前文件版本的审计包。");
    if (!Array.isArray(payload.answers) || payload.answers.length !== wrapper.packet.items.length)
      throw new Error("回答项目数量不一致。");
    const seen = new Set(), drafts = {};
    for (const row of payload.answers) {
      const item = wrapper.packet.items.find(x => x.review_id === row.review_id);
      if (!item || seen.has(row.review_id) || !["sequence_confirmed", "sequence_differs", "pending"].includes(row.status) || row.phone_timing_confirmed !== false)
        throw new Error("回答编号、状态或确认范围不正确。");
      seen.add(row.review_id);
      if (JSON.stringify(stable(row.source_bindings)) !== JSON.stringify(stable(item.source_bindings))) throw new Error("音频或 VSQX 版本不一致。");
      if (row.status === "sequence_confirmed" && row.explicit_sequence_scope_confirmed !== true) throw new Error("完整顺序确认缺少明确勾选。");
      if (typeof row.statement !== "string") throw new Error("补充原话必须是文本。");
      const requested = row.status === "pending" && ["sequence_confirmed", "sequence_differs", "pending"].includes(row.requested_decision) ? row.requested_decision : row.status;
      drafts[row.review_id] = {decision: requested, affirmed: row.explicit_sequence_scope_confirmed === true, statement: row.statement};
    }
    return {drafts, reviewer: String(payload.reviewer || "user")};
  }
  if (typeof module !== "undefined" && module.exports) module.exports = {makeAnswers, parseAnswers};
  if (typeof document === "undefined") return;
  const wrapper = JSON.parse(document.getElementById("packet").textContent);
  const key = "whole-sequence-review:" + wrapper.packet_sha256;
  const message = document.getElementById("message");
  let drafts = {};
  function readControls() {
    wrapper.packet.items.forEach(item => {
      const id = item.review_id;
      const selected = document.querySelector('input[name="' + id + '-decision"]:checked');
      drafts[id] = {decision: selected ? selected.value : "pending", affirmed: document.getElementById(id + "-affirm").checked,
        statement: document.getElementById(id + "-statement").value};
    });
  }
  function updateStatus() {
    const answer = makeAnswers(wrapper, drafts, document.getElementById("reviewer").value);
    answer.answers.forEach(row => {
      document.getElementById(row.review_id + "-status").textContent = row.status === "sequence_confirmed" ? "整曲顺序已确认（不含音素时间）。" : row.status === "sequence_differs" ? "已记录存在差异；原话将完整导出。" : "待定，可随时导出。";
    });
    document.getElementById("progress").textContent = answer.answers.filter(r => r.status !== "pending").length + " / " + answer.answers.length + " 项已作答";
    try { localStorage.setItem(key, JSON.stringify({drafts, reviewer: answer.reviewer})); }
    catch (_) { message.textContent = "浏览器无法自动保存，请及时导出。"; }
  }
  function applyDrafts() {
    wrapper.packet.items.forEach(item => {
      const id = item.review_id, value = drafts[id] || {};
      document.querySelectorAll('input[name="' + id + '-decision"]').forEach(input => {input.checked = input.value === value.decision;});
      document.getElementById(id + "-affirm").checked = value.affirmed === true;
      document.getElementById(id + "-statement").value = value.statement || "";
    });
    updateStatus();
  }
  try { const saved = JSON.parse(localStorage.getItem(key) || "null"); if (saved) {drafts = saved.drafts || {}; document.getElementById("reviewer").value = saved.reviewer || "user";} }
  catch (_) { message.textContent = "自动保存未能读取；可以导入以前导出的 JSON。"; }
  applyDrafts();
  document.querySelectorAll('input[type="radio"],input[type="checkbox"],textarea,#reviewer').forEach(input => input.addEventListener("input", () => {readControls(); updateStatus();}));
  document.querySelectorAll("button[data-play]").forEach(button => button.addEventListener("click", () => {
    const audio = document.getElementById(button.dataset.play);
    document.querySelectorAll("audio").forEach(other => {if (other !== audio) other.pause();});
    audio.currentTime = Number(button.dataset.time);
    const playing = audio.play(); if (playing && playing.catch) playing.catch(err => {message.textContent = "无法播放：" + err.message;});
  }));
  document.getElementById("export").addEventListener("click", () => {
    readControls(); updateStatus();
    const payload = makeAnswers(wrapper, drafts, document.getElementById("reviewer").value);
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], {type: "application/json;charset=utf-8"}));
    const link = document.createElement("a"); link.href = url;
    link.download = "whole-sequence-review-answers-" + wrapper.packet_sha256.slice(0, 12) + ".json";
    link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    message.textContent = "已导出全部回答，待定项和原话均保留。请把 JSON 发回任务。";
  });
  document.getElementById("import").addEventListener("click", () => document.getElementById("import-file").click());
  document.getElementById("import-file").addEventListener("change", async event => {
    const file = event.target.files[0]; if (!file) return;
    try { const data = parseAnswers(wrapper, JSON.parse(await file.text())); drafts = data.drafts;
      document.getElementById("reviewer").value = data.reviewer; applyDrafts(); message.textContent = "已导入当前审计包的回答。";
    } catch (err) {message.textContent = err.message;}
    event.target.value = "";
  });
})();
