const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const {makeAnswers, parseAnswers} = require('../tools/whole_sequence_review_state.js');
const wrapper = {packet_sha256:'a'.repeat(64),packet:{items:[
  {review_id:'S01',source_bindings:{original_audio:{path:'original.wav',sha256:'b'.repeat(64)},human_vsqx:{path:'song.vsqx',sha256:'c'.repeat(64)}}},
  {review_id:'S02',source_bindings:{original_audio:{path:'other.wav',sha256:'d'.repeat(64)}}}
]}};
test('all pending answers export without any required choices',()=>{
  const result=makeAnswers(wrapper,{},'user');
  assert.deepEqual(result.answers.map(x=>x.status),['pending','pending']);
  assert.ok(result.answers.every(x=>x.phone_timing_confirmed===false));
  assert.equal(parseAnswers(wrapper,result).drafts.S01.decision,'pending');
});
test('unchecked scope remains pending and retains raw words and draft choice',()=>{
  const result=makeAnswers(wrapper,{S01:{decision:'sequence_confirmed',affirmed:false,statement:'不确定 </script>'}},'user');
  assert.equal(result.answers[0].status,'pending');
  assert.equal(result.answers[0].statement,'不确定 </script>');
  const loaded=parseAnswers(wrapper,result);
  assert.equal(loaded.drafts.S01.decision,'sequence_confirmed');
  assert.equal(loaded.drafts.S01.affirmed,false);
});
test('confirmed order never claims timing and reordered source keys import',()=>{
  const result=makeAnswers(wrapper,{S01:{decision:'sequence_confirmed',affirmed:true}},'user');
  const binding=result.answers[0].source_bindings;
  result.answers[0].source_bindings={human_vsqx:{sha256:binding.human_vsqx.sha256,path:binding.human_vsqx.path},original_audio:binding.original_audio};
  assert.equal(parseAnswers(wrapper,result).drafts.S01.decision,'sequence_confirmed');
  assert.equal(result.answers[0].phone_timing_confirmed,false);
});
test('wrong packet, changed sources, duplicate rows and invented timing rejected',()=>{
  for (const mutate of [r=>r.packet_sha256='e'.repeat(64),r=>r.answers[0].source_bindings={},
    r=>r.answers[1]=r.answers[0],r=>r.answers[0].phone_timing_confirmed=true]) {
    const result=JSON.parse(JSON.stringify(makeAnswers(wrapper,{},'user')));mutate(result);
    assert.throws(()=>parseAnswers(wrapper,result));
  }
});
test('DOM input preserves page nodes/scroll and enables pending export',()=>{
  const elements={};
  function element(id){return elements[id]={id,value:'',checked:false,textContent:'',listeners:{},addEventListener(n,f){this.listeners[n]=f;},click(){this.clicked=true;}};}
  element('packet').textContent=JSON.stringify(wrapper);
  for (const id of ['message','progress','reviewer','export','import','import-file']) element(id);
  elements.reviewer.value='user';
  const controls=[];
  for (const item of wrapper.packet.items) {
    for (const value of ['sequence_confirmed','sequence_differs','pending']) {const e=element(item.review_id+'-'+value);e.name=item.review_id+'-decision';e.value=value;controls.push(e);}
    controls.push(element(item.review_id+'-affirm'));controls.push(element(item.review_id+'-statement'));element(item.review_id+'-status');
  }
  controls.push(elements.reviewer);
  const doc={getElementById:id=>elements[id],querySelector(selector){const name=selector.match(/name="([^"]+)"/)[1];return controls.find(e=>e.name===name&&e.checked)||null;},
    querySelectorAll(selector){if(selector.startsWith('input[name=')){const name=selector.match(/name="([^"]+)"/)[1];return controls.filter(e=>e.name===name);}if(selector.startsWith('input[type='))return controls;return [];},
    createElement(){return {click(){this.clicked=true;}};}};
  const store=new Map();let downloads=0;
  const context={document:doc,localStorage:{getItem:k=>store.get(k)||null,setItem:(k,v)=>store.set(k,v)},
    URL:{createObjectURL:()=>{downloads++;return 'blob:fixture';},revokeObjectURL(){}},Blob,Date,JSON,Set,Object,
    setTimeout:()=>0,window:{scrollY:800,scrollTo(){throw Error('unexpected scroll');}}};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../tools/whole_sequence_review_state.js'),'utf8'),context);
  const originalNodes=Object.values(elements);
  elements['S01-sequence_confirmed'].checked=true;elements['S01-sequence_confirmed'].listeners.input();
  assert.match(elements['S01-status'].textContent,/待定/);
  elements['S01-affirm'].checked=true;elements['S01-affirm'].listeners.input();
  assert.match(elements['S01-status'].textContent,/已确认/);
  elements.export.listeners.click();assert.equal(downloads,1);
  assert.deepEqual(Object.values(elements),originalNodes);assert.equal(context.window.scrollY,800);
});
