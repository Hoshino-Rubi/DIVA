"use strict";
const {test} = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs"), vm = require("node:vm"), path = require("node:path");
vm.runInThisContext(fs.readFileSync(path.join(__dirname,"../tools/alignment_audit_gui/audit_core.js"),"utf8"));
function bundle() {
  return {bundle_id:"a".repeat(64),sources:{reference:{sha256:"f".repeat(64)}},rows:[
    {review_id:"C01",packet_sha256:"b".repeat(64),song_id:"like-you",song_name:"喜欢你",kind:"training",
      human:{status:"rejected",review_statement:"错位"},state:{human_rejected:true,human_machine_conflict:true},notes:[{lyric:"笑"}]},
    {review_id:"B01",packet_sha256:"c".repeat(64),song_id:"shalong",song_name:"沙龙",kind:"boundary",
      human:{status:"pending",review_statement:"框住目前"},state:{human_rejected:false,machine_flagged:true},notes:[{lyric:"框"}]},
  ]};
}
function draft(extra={}) { return {decision:"uncertain",boundary_state:"unknown",reviewer:"Leo",first_heard:"框",last_heard:"前",comment:"只确认对应，不猜音素 ŋ",...extra}; }
const make = () => AlignmentAudit.createStore(bundle());
test("initial audit has no fabricated confirmations",()=>{
  const s=make(), e=s.exportData(); assert.equal(s.count(),0); assert.equal(e.decisions.length,0);
  assert.equal(e.creates_phone_reference,false);assert.equal(e.starts_training,false);
});
test("human rejection hidden by default but visible in conflicts",()=>{
  const s=make(), b=bundle();
  assert.equal(s.matches(b.rows[0],"attention","",""),false);
  assert.equal(s.matches(b.rows[0],"conflict","",""),true);
  assert.equal(s.matches(b.rows[1],"attention","",""),true);
  assert.equal(s.matches(b.rows[1],"boundary","shalong","框"),true);
});
test("save and export/import preserve Unicode and exact source identity",()=>{
  const s=make();s.save("B01",draft());const e=JSON.parse(JSON.stringify(s.exportData()));
  const t=make();assert.equal(t.importData(e),1);assert.deepEqual(t.get("B01"),s.get("B01"));
  assert.equal(t.get("B01").comment,"只确认对应，不猜音素 ŋ");
});
test("rejection override needs explicit acknowledgement",()=>{
  const s=make();assert.throws(()=>s.save("C01",draft({decision:"corresponds"})),/明确勾选/);
  s.save("C01",draft({decision:"corresponds",acknowledge_rejection_override:true}));
  assert.equal(s.get("C01").creates_phone_reference,false);
});
test("unknown IDs, missing reviewer and invalid choices are rejected",()=>{
  const s=make();assert.throws(()=>s.save("C99",draft()),/未知/);
  assert.throws(()=>s.save("B01",draft({reviewer:" "})),/复核人/);
  assert.throws(()=>s.save("B01",draft({decision:"human_confirmed"})),/未知/);
  assert.equal(s.count(),0);
});
test("bad bundle/source hashes and phone-truth promotion cannot be imported",()=>{
  for(const mutate of [e=>e.bundle_id="d".repeat(64),e=>e.source_hashes.reference="0".repeat(64),e=>e.creates_phone_reference=true]){
    const s=make();s.save("B01",draft());const e=s.exportData();mutate(e);assert.throws(()=>make().importData(e));
  }
});
test("duplicate or invalid import is atomic; existing records survive",()=>{
  const s=make();s.save("B01",draft());const e=s.exportData();e.decisions.push({...e.decisions[0]});
  const t=make();assert.throws(()=>t.importData(e),/重复/);assert.equal(t.count(),0);
  t.save("C01",draft({decision:"misaligned"}));e.decisions[1].review_id="unknown";
  assert.throws(()=>t.importData(e));assert.equal(t.count(),1);assert.equal(t.get("B01"),null);
});
test("clearing an audit does not alter immutable human source or other items",()=>{
  const b=bundle(), original=JSON.stringify(b), s=AlignmentAudit.createStore(b);
  s.save("C01",draft({decision:"misaligned"}));s.save("B01",draft());s.clear("C01");
  assert.equal(s.count(),1);assert.equal(JSON.stringify(b),original);
  const returned=s.get("B01");returned.decision="human_confirmed";assert.equal(s.get("B01").decision,"uncertain");
});
test("saved rows disappear from attention and appear in saved filter",()=>{
  const b=bundle(),s=AlignmentAudit.createStore(b);s.save("B01",draft());
  assert.equal(s.matches(b.rows[1],"attention","",""),false);
  assert.equal(s.matches(b.rows[1],"saved","",""),true);
});
