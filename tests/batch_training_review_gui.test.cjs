'use strict';
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const S=require('../tools/batch_training_review_gui/state.js');
function fixture(){
 const media={audio_file:'edge.wav',waveform_file:'edge.png'};
 const notes=[{note_id:'n0',lyric:'<script>字</script>',phones:['n','a'],mapping_status:'resolved_proxy'},
 {note_id:'n1',lyric:'末',phones:['m','a'],mapping_status:'resolved_proxy'}];
 const choice={choice_id:'A',start_sec:1,end_sec:11,preview:media,notes,catalog:notes,edge_notes:{start:notes,end:notes},edge_score_gaps:[],mapping_complete:true};
 return {schema_version:'batch_training_review_packet_v1',packet_sha256:'a'.repeat(64),items:[{
  review_id:'R01',song_id:'promise',kind:'replacement',start_sec:10,end_sec:20,original:media,edge_media:{start:media,end:media},
  alternatives:[choice,{...structuredClone(choice),choice_id:'B'}],correspondence_preconfirmed:false,previous_audit:null}]};
}
function fullDraft(item){const d=S.empty(item);d.choice='A';d.status='confirmed';d.sequence_confirmed=true;
 for(const [side,note] of [['start','n0'],['end','n1']])d.edges[side]={note_id:note,phones:[{phone_index:0,edge_status:'absent'},{phone_index:1,edge_status:'full'}]};return d;}
test('previous uncertain edges restore as draft only and switching choice clears them',()=>{
 const p=fixture(),item=p.items[0],d=fullDraft(item);
 d.edges.start.phones[1].edge_status='uncertain';
 item.previous_answer=S.commit(p,item,d,'human');
 const restored=S.empty(item);
 assert.equal(restored.edges.start.phones[1].edge_status,'uncertain');
 assert.equal(restored.status,'');assert.equal(restored.sequence_confirmed,false);
 assert.throws(()=>S.commit(p,item,restored,'human'));
 restored.edges.start.phones[1].edge_status='full';
 assert.equal(item.previous_answer.edges.start.phones[1].edge_status,'uncertain');
 assert.deepEqual(S.setChoice(item,restored,'B').edges.start.phones,[]);
});
test('new choices and phones are empty, existing correspondence alone is inherited',()=>{
 const p=fixture();assert.equal(S.empty(p.items[0]).choice,'');
 const old={...p.items[0],correspondence_preconfirmed:true};const d=S.empty(old);assert.equal(d.choice,'A');assert.equal(d.edges.start.note_id,null);assert.equal(d.sequence_confirmed,false);
 const changed=S.setChoice(p.items[0],fullDraft(p.items[0]),'B');assert.deepEqual(changed.edges.start.phones,[]);assert.equal(changed.sequence_confirmed,false);
});
test('combined answer validates roundtrip, partial and absent do not get recoded',()=>{
 const p=fixture(),d=fullDraft(p.items[0]);d.edges.end.phones[1].edge_status='partial';
 const a=S.commit(p,p.items[0],d,'human');const out=S.exportAnswers(p,[a],'human');
 assert.equal(out.schema_version,'batch_training_review_answers_v1');assert.equal(out.answers[0].edges.start.phones[0].edge_status,'absent');
 assert.equal(S.draftFromAnswer(p.items[0],a).edges.end.phones[1].edge_status,'partial');
});
test('wrong packet, missing phone and invalid prefix ordering fail; uncertain is preserved',()=>{
 const p=fixture(),d=fullDraft(p.items[0]);d.edges.start.phones.pop();assert.throws(()=>S.commit(p,p.items[0],d,'human'),/每个/);
 d.edges.start.phones.push({phone_index:1,edge_status:'uncertain'});const a=S.commit(p,p.items[0],d,'human');assert.equal(a.edges.start.phones[1].edge_status,'uncertain');
 const out=S.exportAnswers(p,[a],'human');assert.throws(()=>S.validate(p,{...out,packet_sha256:'b'.repeat(64)}),/不匹配/);
 assert.throws(()=>S.validate(p,{...out,answers:[a,a]}),/重复/);
});
test('reject or uncertain submissions have no automatic phone assertions',()=>{
 const p=fixture();for(const status of ['rejected','uncertain']){const d=fullDraft(p.items[0]);d.status=status;const a=S.commit(p,p.items[0],d,'human');assert.equal(a.choice,null);assert.deepEqual(a.edges,{});}
});

function mount(storage=new Map()){
 const viewport={scrollX:0,scrollY:750};
 class Node{
  constructor(tag){this.tag=tag;this.children=[];this.events={};this.textContent='';this.value='';}
  append(...nodes){this.children.push(...nodes);}replaceChildren(...nodes){this.children=nodes;}
  addEventListener(name,fn){this.events[name]=fn;}setAttribute(name,value){this[name]=value;}
  click(){return this.events.click?.();}remove(){}pause(){}
 }
 const packet=fixture(),ids=Object.fromEntries(['packet','audio-data','reviewer','notice','progress','navigation','review','save','export','import'].map(id=>[id,new Node()]));
 ids.packet.textContent=JSON.stringify(packet);ids['audio-data'].textContent=JSON.stringify({'edge.wav':'AA=='});
 const flatten=n=>[n,...n.children.flatMap(flatten)],nodes=()=>flatten(ids.review);
 const starts=[],exports=[];
 class AudioContext{async decodeAudioData(){return {duration:2.5};}async resume(){}createBufferSource(){return {connect(){},start(...args){starts.push(args);},stop(){}};}}
 const win={BatchReviewState:S,AudioContext,confirm:()=>true,scrollTo:({left,top})=>{viewport.scrollX=left;viewport.scrollY=top;}};
 Object.defineProperties(win,{scrollX:{get:()=>viewport.scrollX},scrollY:{get:()=>viewport.scrollY}});
 const sandbox={window:win,document:{getElementById:id=>ids[id]||nodes().find(n=>n.id===id),createElement:tag=>new Node(tag),
  createTextNode:text=>Object.assign(new Node('text'),{textContent:text}),body:new Node('body'),querySelectorAll:tag=>nodes().filter(n=>n.tag===tag)},
  localStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v)},structuredClone,Uint8Array,atob:()=>String.fromCharCode(0),
  URL:{createObjectURL:()=> 'blob:fixture',revokeObjectURL(){}},Blob:class{constructor(parts){exports.push(JSON.parse(parts[0]));}},setTimeout(){}};
 vm.runInNewContext(fs.readFileSync('tools/batch_training_review_gui/app.js','utf8'),sandbox);
 return {ids,nodes,packet,storage,starts,exports,viewport};
}
test('DOM same segment covers alternative and both edges, phone clicks retain form and scroll',async()=>{
 const g=mount();g.ids.reviewer.value='human';g.ids.reviewer.events.input();
 assert.ok(g.nodes().filter(n=>n.type==='radio').every(n=>!n.checked));
 g.nodes().find(n=>n.type==='radio'&&n.value==='A').events.change();
 let selectors=g.nodes().filter(n=>n.tag==='select'&&['首字','末字'].includes(n['aria-label']));
 selectors[0].value='n0';selectors[0].events.change();selectors=g.nodes().filter(n=>n.tag==='select'&&['首字','末字'].includes(n['aria-label']));selectors[1].value='n1';selectors[1].events.change();
 const originals=[...g.ids.review.children];const bulk=g.nodes().filter(n=>n.tag==='button'&&n.textContent==='本字全部完整可听');bulk[0].click();bulk[1].click();
 const phone=g.nodes().find(n=>n.tag==='button'&&n.textContent==='完全听不到');const root=g.ids.review.children[0];phone.click();
 assert.equal(g.ids.review.children[0],root);assert.equal(g.viewport.scrollY,750);
 const status=g.nodes().find(n=>n['aria-label']==='本段结论');status.value='confirmed';status.events.change();
 const box=g.nodes().find(n=>n.id==='affirm');box.checked=true;box.events.change();
 await g.nodes().find(n=>n.tag==='button'&&n.textContent.includes('听右半')).click();await g.nodes().find(n=>n.tag==='button'&&n.textContent.includes('听左半')).click();
 assert.deepEqual(g.starts,[[0,1.25,1.25],[0,0,1.25]]);
 g.nodes().find(n=>n.tag==='button'&&n.textContent==='提交本段，下一段').click();g.ids.export.click();
 assert.equal(g.exports.length,1);assert.equal(g.exports[0].answers[0].edges.start.phones[0].edge_status,'absent');
 assert.ok(originals.length>0);const restored=mount(g.storage);restored.ids.export.click();assert.deepEqual(restored.exports[0].answers,g.exports[0].answers);
});
test('invalid import preserves current draft and markup-like lyrics are text only',async()=>{
 const g=mount();assert.ok(g.nodes().some(n=>n.textContent.includes('<script>字</script>')));
 const event={target:{files:[{text:async()=>JSON.stringify({schema_version:'wrong',packet_sha256:'b'.repeat(64)})}],value:'x'}};
 await g.ids.import.events.change(event);assert.match(g.ids.notice.textContent,/导入失败/);
 assert.ok(g.nodes().filter(n=>n.type==='radio').every(n=>!n.checked));
 const code=fs.readFileSync('tools/batch_training_review_gui/app.js','utf8');assert.doesNotMatch(code,/innerHTML|fetch\(|autoplay/);
});
