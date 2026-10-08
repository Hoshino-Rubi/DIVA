/* Pure review state. Deliberately not the human phone-reference v2 schema. */
(function (root) {
  "use strict";
  const SCHEMA = "alignment_audit_decisions_v1";
  const choices = new Set(["corresponds", "misaligned", "boundary_issue", "uncertain"]);
  const boundaries = new Set(["unknown", "full", "truncated_start", "truncated_end", "truncated_both"]);
  const clone = value => JSON.parse(JSON.stringify(value));
  const text = (value, limit, label) => {
    if (typeof value !== "string" || value.length > limit) throw Error(label + "格式或长度不正确");
    return value;
  };
  function createStore(bundle) {
    const byId = new Map(bundle.rows.map(row => [row.review_id, row]));
    let decisions = new Map();
    function validate(raw) {
      if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw Error("审计记录格式不正确");
      const item = byId.get(raw.review_id);
      if (!item || raw.packet_sha256 !== item.packet_sha256) throw Error("复核编号或来源哈希不匹配");
      if (!choices.has(raw.decision) || !boundaries.has(raw.boundary_state)) throw Error("未知审计状态");
      const reviewer = text(raw.reviewer, 120, "复核人");
      if (!reviewer.trim()) throw Error("请填写复核人");
      if (item.state.human_rejected && raw.decision === "corresponds" && raw.acknowledge_rejection_override !== true)
        throw Error("此项原先判为错位；修改为对应正确前，需要明确勾选修订确认");
      if (typeof raw.updated_at !== "string" || !Number.isFinite(Date.parse(raw.updated_at))) throw Error("记录时间无效");
      return {
        review_id: item.review_id, packet_sha256: item.packet_sha256, decision: raw.decision,
        boundary_state: raw.boundary_state, reviewer,
        first_heard: text(raw.first_heard, 160, "开头字"), last_heard: text(raw.last_heard, 160, "结尾字"),
        comment: text(raw.comment, 4000, "说明"), updated_at: raw.updated_at,
        acknowledge_rejection_override: raw.acknowledge_rejection_override === true,
        scope: "correspondence_audit_only", creates_phone_reference: false,
      };
    }
    function save(id, draft) {
      const item = byId.get(id);
      if (!item) throw Error("未知复核编号");
      const value = validate({...draft, review_id: id, packet_sha256: item.packet_sha256, updated_at: new Date().toISOString()});
      decisions.set(id, value);
      return clone(value);
    }
    function exportData() {
      return {
        schema_version: SCHEMA, bundle_id: bundle.bundle_id, exported_at: new Date().toISOString(),
        source_hashes: Object.fromEntries(Object.entries(bundle.sources).map(([key, value]) => [key, value.sha256])),
        decisions: bundle.rows.filter(row => decisions.has(row.review_id)).map(row => clone(decisions.get(row.review_id))),
        automatic_annotation_updates_allowed: false, creates_phone_reference: false, starts_training: false,
      };
    }
    function importData(payload) {
      if (!payload || payload.schema_version !== SCHEMA) throw Error("不是本界面的审计文件；旧回答已经显示在历史意见中");
      if (payload.bundle_id !== bundle.bundle_id) throw Error("文件属于另一版审计数据，不能混合导入");
      if (payload.creates_phone_reference !== false || payload.automatic_annotation_updates_allowed !== false || payload.starts_training !== false)
        throw Error("文件混入了音素确认或训练授权，已拒绝导入");
      for (const [key, value] of Object.entries(bundle.sources)) {
        if (payload.source_hashes?.[key] !== value.sha256) throw Error("审计来源哈希不匹配");
      }
      if (!Array.isArray(payload.decisions) || payload.decisions.length > byId.size) throw Error("审计记录数量不正确");
      const staged = new Map();
      for (const raw of payload.decisions) {
        const value = validate(raw);
        if (staged.has(value.review_id)) throw Error("文件含重复复核编号");
        staged.set(value.review_id, value);
      }
      for (const [id, value] of staged) decisions.set(id, value);
      return staged.size;
    }
    function matches(row, mode, song, query) {
      if (song && row.song_id !== song) return false;
      if (mode === "attention" && (row.state.human_rejected || decisions.has(row.review_id))) return false;
      if (mode === "conflict" && !row.state.human_machine_conflict) return false;
      if (mode === "rejected" && !row.state.human_rejected) return false;
      if (mode === "machine_issue" && !row.state.machine_flagged) return false;
      if (mode === "boundary" && row.kind !== "boundary") return false;
      if (mode === "saved" && !decisions.has(row.review_id)) return false;
      const q = query.trim().toLocaleLowerCase();
      return !q || [row.review_id, row.song_name, row.human.review_statement || "", ...row.notes.map(n => n.lyric)].join(" ").toLocaleLowerCase().includes(q);
    }
    return {save, exportData, importData, matches,
      get: id => decisions.has(id) ? clone(decisions.get(id)) : null,
      clear: id => decisions.delete(id), count: () => decisions.size,
    };
  }
  root.AlignmentAudit = {createStore, SCHEMA};
})(globalThis);
