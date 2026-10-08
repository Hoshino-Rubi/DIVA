'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const S=require('../tools/free_location_review_gui/state.js');
function fixture(){
 const media=(name,duration)=>({audio_file:name,duration_sec:duration});
 return {schema_version:'free_location_review_packet_v1',packet_sha256:'a'.repeat(64),song_name:'fixture',
  windows:['A','B'].map((v,i)=>({window_id:'M01-'+v,start_sec:10+i*20,end_sec:20+i*20,original:media('original.wav',10),edges:{start:media('edge.wav',2.5),end:media('edge.wav',2.5)}})),
  catalog:['甲','乙','丙','丁','甲','乙'].map((lyric,i)=>({note_id:'n'+i,lyric,phones:['a','n'],mapping_status:'resolved_proxy',start_sec:i,end_sec:i+1})),
  rendered_preview:media('rendered.wav',20)};
}
function filled(p){const d=S.empty(p);d.status='confirmed';d.sequence_confirmed=true;for(const [side,id] of [['start','n0'],['end','n3']])d.edges[side]={note_id:id,phones:[{phone_index:0,edge_status:'absent'},{phone_index:1,edge_status:'partial'}]};return d;}
test('whole catalog remains selectable, no initial guessed notes, duplicate lyrics stay distinct',()=>{
 const p=fixture(),d=S.empty(p);assert.equal(d.edges.start.note_id,null);assert.equal(d.edges.end.note_id,null);
 const ids=S.searchIndices(p,'甲乙');assert.ok(ids.includes(0)&&ids.includes(4));
 const a=S.exportAnswer(p,filled(p),'human');assert.equal(a.answer.edges.start.phones[0].edge_status,'absent');assert.equal(a.answer.edges.end.phones[1].edge_status,'partial');
 assert.equal(S.selected(p,a.answer).length,4);
});
test('invalid and incomplete references fail without inventing phones',()=>{
 const p=fixture(),d=filled(p);d.edges.start.phones.pop();assert.throws(()=>S.exportAnswer(p,d,'human'),/每个音素/);
 d.edges.start.phones.push({phone_index:1,edge_status:'partial'});p.catalog[1].mapping_status='unresolved';assert.throws(()=>S.exportAnswer(p,d,'human'),/未解决/);
 d.status='pending';const out=S.exportAnswer(p,d,'human');assert.deepEqual(out.answer.edges,{});assert.equal(out.unconfirmed_draft.edges.start.note_id,'n0');
 assert.deepEqual(S.fromPayload(p,out).edges,d.edges);assert.equal(S.fromPayload(p,out).sequence_confirmed,false);
});
test('packet mismatch and trying multiple answers fail',()=>{
 const p=fixture(),out=S.exportAnswer(p,filled(p),'human');assert.throws(()=>S.validate(p,{...out,packet_sha256:'b'.repeat(64)}));
 assert.throws(()=>S.validate(p,{...out,answers:[out.answer,out.answer]}));
});
function mount(storage=new Map()){
 const viewport={scrollX:0,scrollY:800},flatten=n=>[n,...n.children.flatMap(flatten)];
 class Node{
  constructor(tag){this.tag=tag;this.children=[];this.events={};this.dataset={};this.textContent='';this.value='';}
  append(...n){this.children.push(...n);}replaceChildren(...n){this.children=n;}addEventListener(e,f){this.events[e]=f;}
  click(){return this.events.click?.();}querySelectorAll(q){return flatten(this).filter(n=>q==='[data-note-id]'&&n.dataset.noteId);}
 }
 const names=['packet','audio-data','reviewer','notice','save','import','windows','original','search','clear-search','play-range','stop','search-count','catalog','range','start-edge','end-edge','status','affirm','comment','export'];
 const ids=Object.fromEntries(names.map(id=>[id,new Node('div')])),packet=fixture(),exports=[],starts=[];
 ids.packet.textContent=JSON.stringify(packet);ids['audio-data'].textContent=JSON.stringify({'original.wav':'Cg==','edge.wav':'GQ==','rendered.wav':'FA=='});
 class AudioContext{async decodeAudioData(bytes){const value=new Uint8Array(bytes)[0];return {duration:value===25?2.5:value};}async resume(){}createBufferSource(){return {connect(){},stop(){},start(...args){starts.push(args);}};}}
 const win={FreeLocationState:S,AudioContext,confirm:()=>true,scrollTo:({left,top})=>{viewport.scrollX=left;viewport.scrollY=top;}};
 Object.defineProperties(win,{scrollX:{get:()=>viewport.scrollX},scrollY:{get:()=>viewport.scrollY}});
 vm.runInNewContext(fs.readFileSync('tools/free_location_review_gui/app.js','utf8'),{window:win,document:{getElementById:id=>ids[id],createElement:tag=>new Node(tag)},localStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v)},structuredClone,Uint8Array,atob:s=>Buffer.from(s,'base64').toString('binary'),URL:{createObjectURL:()=> 'blob:test',revokeObjectURL(){}},Blob:class{constructor(parts){exports.push(JSON.parse(parts[0]));}},setTimeout(){}});
 const nodes=id=>flatten(ids[id]);return {ids,nodes,exports,starts,storage,viewport,packet};
}
test('DOM full selection, changing one edge keeps other handlers live, export and restore',async()=>{
 const g=mount();g.ids.reviewer.value='human';g.ids.reviewer.events.input();
 const pick=(id,text)=>g.nodes('catalog').find(n=>n.dataset.noteId===id).children.find(n=>n.tag==='button'&&n.textContent===text).click();
 pick('n0','设为首字');pick('n3','设为末字');pick('n1','设为首字');
 for(const side of ['start','end'])g.nodes(side+'-edge').find(n=>n.tag==='button'&&n.textContent==='本字全部完整可听').click();
 g.nodes('end-edge').find(n=>n.tag==='button'&&n.textContent==='完全听不到').click();
 assert.equal(g.viewport.scrollY,800);
 g.ids.status.value='confirmed';g.ids.status.events.change();g.ids.affirm.checked=true;g.ids.affirm.events.change();
 await g.nodes('original').find(n=>n.tag==='button'&&n.textContent.includes('十秒')).click();
 await g.nodes('start-edge').find(n=>n.tag==='button'&&n.textContent.includes('只听切点后')).click();
 await g.nodes('end-edge').find(n=>n.tag==='button'&&n.textContent.includes('只听切点前')).click();
 assert.deepEqual(g.starts,[[0,0,10],[0,1.25,1.25],[0,0,1.25]]);
 g.ids.export.click();assert.equal(g.exports.length,1);const a=g.exports[0].answer;
 assert.equal(a.edges.start.note_id,'n1');assert.equal(a.edges.end.phones[0].edge_status,'absent');
 const restored=mount(g.storage);restored.ids.export.click();assert.deepEqual(restored.exports[0].answer,a);
});
test('switching originals clears choices; bad imports preserve work; no network or unsafe HTML',async()=>{
 const g=mount();g.ids.reviewer.value='human';g.ids.status.value='pending';g.ids.status.events.change();
 await g.ids.import.events.change({target:{files:[{text:async()=>JSON.stringify({schema_version:'wrong'})}],value:'x'}});assert.match(g.ids.notice.textContent,/导入失败/);
 g.nodes('windows').find(n=>n.tag==='button'&&n.textContent.includes('备用')).click();assert.equal(g.ids.status.value,'');
 assert.ok(g.ids.range.textContent.includes('尚未'));
 const code=fs.readFileSync('tools/free_location_review_gui/app.js','utf8');assert.doesNotMatch(code,/innerHTML|fetch\(|autoplay/);
});
