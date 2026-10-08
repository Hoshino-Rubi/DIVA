'use strict';
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const html=fs.readFileSync('tools/replacement_location_gui.html','utf8');
const code=[...html.matchAll(/<script(?: [^>]*)?>([\s\S]*?)<\/script>/g)].at(-1)[1];
function mount(storage=new Map()){
 class Node{
  constructor(tag){this.tag=tag;this.children=[];this.events={};this.textContent='';this.value='';}
  append(...nodes){this.children.push(...nodes);}
  replaceChildren(...nodes){this.children=nodes;}
  addEventListener(type,fn){this.events[type]=fn;}
  setAttribute(){}
  click(){this.events.click?.();}
  remove(){}
 }
 const media={audio_file:'x.wav',waveform_file:'x.png'};
 const packet={packet_sha256:'a'.repeat(64),items:[{review_id:'L01',song_id:'promise',start_sec:2,end_sec:12,
  original:media,alternatives:[{choice_id:'A',start_sec:4,end_sec:14,preview:media,mapping_complete:true,
   notes:[{raw_lyric:'<script>unsafe</script>'}]}]}]};
 const ids=Object.fromEntries(['packet','reviewer','items','navigation','notice','export','import'].map(k=>[k,new Node()]));
 ids.packet.textContent=JSON.stringify(packet);
 const flatten=n=>[n,...n.children.flatMap(flatten)];
 const nodes=()=>flatten(ids.items);
 const payloads=[];
 const sandbox={document:{getElementById:id=>ids[id]||nodes().find(n=>n.id===id),createElement:tag=>new Node(tag),
  createTextNode:text=>Object.assign(new Node('text'),{textContent:text}),body:new Node('body')},
  localStorage:{getItem:key=>storage.get(key),setItem:(key,value)=>storage.set(key,value)},window:{confirm:()=>true},
  URL:{createObjectURL:()=> 'blob:fixture',revokeObjectURL(){}},Blob:class{constructor(parts){payloads.push(JSON.parse(parts[0]));}},setTimeout(){}};
 vm.runInNewContext(code,sandbox);
 return {ids,nodes,storage,payloads,packet};
}
test('empty choices, text-safe rendering, no radio re-render, and bound export',()=>{
 const gui=mount();
 assert.ok(gui.nodes().filter(n=>n.type==='radio').every(n=>!n.checked));
 assert.ok(gui.nodes().some(n=>n.textContent==='<script>unsafe</script>'));
 gui.ids.export.click();assert.equal(gui.payloads.length,0);
 gui.ids.reviewer.value='fixture';gui.ids.reviewer.events.input();
 const art=gui.ids.items.children[0];
 const radios=gui.nodes().filter(n=>n.type==='radio');
 radios[1].events.change();
 assert.equal(gui.ids.items.children[0],art);
 const text=gui.nodes().find(n=>n.tag==='textarea');text.value='only part corresponds';text.events.input();
 gui.ids.export.click();
 assert.equal(gui.payloads[0].answers[0].choice,'none');
 assert.equal(gui.payloads[0].answers[0].comment,'only part corresponds');
 assert.equal(gui.payloads[0].packet_sha256,gui.packet.packet_sha256);
 assert.equal(gui.payloads[0].training_started,false);
 const restored=mount(gui.storage);restored.ids.export.click();
 assert.deepEqual(restored.payloads[0].answers,gui.payloads[0].answers);
});
test('import refuses wrong packet without changing existing answers',async()=>{
 const gui=mount();
 const event={target:{files:[{text:async()=>JSON.stringify({schema_version:'replacement_location_answers_v1',packet_sha256:'b'.repeat(64),answers:[],reviewer:'fixture'})}],value:'file'}};
 await gui.ids.import.events.change(event);
 assert.match(gui.ids.notice.textContent,/不匹配/);
 assert.ok(gui.nodes().filter(n=>n.type==='radio').every(n=>!n.checked));
});
test('no network, prediction scores, automatic media playback or markup injection',()=>{
 assert.doesNotMatch(code,/fetch\(|innerHTML|autoplay/);
 assert.doesNotMatch(html,/PER|seed43/);
});
