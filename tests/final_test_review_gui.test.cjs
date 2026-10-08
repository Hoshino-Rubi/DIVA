'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const F=require('../tools/final_test_review_gui/state.js');
function fixture(){
 const media=(name,duration)=>({audio_file:name,duration_sec:duration});
 const catalog=['甲','乙','丙','丁'].map((lyric,i)=>({note_id:'n'+i,lyric,phones:['a','n'],mapping_status:'resolved_proxy',raw_v4c_tokens:['a','n'],start_sec:i,end_sec:i+1}));
 return {schema_version:'frozen_final_test_review_packet_v1',packet_sha256:'a'.repeat(64),
  songs:[{song_id:'s',song_name:'歌曲',catalog,rendered_preview:media('reference.wav',20)}],
  items:Array.from({length:6},(_,i)=>({window_id:'T0'+(i+1),song_id:'s',start_sec:i*20+1,end_sec:i*20+19,original:media('original.wav',18),edges:{start:media('edge.wav',2.5),end:media('edge.wav',2.5)}})),
  mapping_questions:['7','UN','iE_r','iUN','uo'].map(token=>({token,contexts:[{song_id:'s',song_name:'歌曲',lyric:'甲',raw_v4c_tokens:[token],start_sec:1,end_sec:2}]}))};
}
function pending(p){const d=F.blank(p);for(const x of Object.values(d.answers))x.status='pending';return d;}

test('six answers exported once, no initial machine choices',()=>{const p=fixture(),d=pending(p);assert.equal(d.answers.T01.edges.start.note_id,null);const a=F.exportAll(p,d,'human');assert.equal(a.answers.length,6);assert.equal(a.mapping_responses.length,5);assert.ok(a.answers.every(x=>!x.sequence_confirmed));assert.deepEqual(F.restore(p,a),d);});
test('located-only preserves missing mapping and never creates phone labels',()=>{const p=fixture(),d=pending(p),x=d.answers.T01;x.status='located';x.sequence_confirmed=true;x.edges.start={note_id:'n0',phones:[{phone_index:0,edge_status:'partial'}]};x.edges.end={note_id:'n3',phones:[],whole_note_audibility:'full'};const a=F.exportAll(p,d,'human');assert.deepEqual(a.answers[0].edges.start.phones,[]);assert.equal(a.unconfirmed_drafts.T01.edges.start.phones[0].edge_status,'partial');assert.deepEqual(F.restore(p,a).answers.T01.edges,x.edges);});
test('mismatched or duplicate windows fail without replacing drafts',()=>{const p=fixture(),a=F.exportAll(p,pending(p),'human');assert.throws(()=>F.validate(p,{...a,packet_sha256:'bad'}));a.answers[1]=a.answers[0];assert.throws(()=>F.validate(p,a));});
function mount(storage=new Map(),packet=fixture()){
 const viewport={scrollX:0,scrollY:750},flatten=n=>[n,...n.children.flatMap(flatten)];
 class Node{constructor(tag){this.tag=tag;this.children=[];this.events={};this.dataset={};this.textContent='';this.value='';}append(...n){this.children.push(...n);}replaceChildren(...n){this.children=n;}addEventListener(e,f){this.events[e]=f;}click(){return this.events.click?.();}querySelectorAll(q){return flatten(this).filter(n=>q==='[data-note-id]'&&n.dataset.noteId);}}
 const names=['packet','audio-data','reviewer','notice','save','import','windows','original','search','clear-search','play-range','stop','search-count','catalog','range','start-edge','end-edge','status','affirm','comment','export','mapping','mapping-status'];
 const ids=Object.fromEntries(names.map(id=>[id,new Node('div')])),exports=[],starts=[];
 ids.packet.textContent=JSON.stringify(packet);ids['audio-data'].textContent=JSON.stringify({'original.wav':'Eg==','edge.wav':'GQ==','reference.wav':'FA=='});
 class AudioContext{async decodeAudioData(bytes){const v=new Uint8Array(bytes)[0];return {duration:v===25?2.5:v};}async resume(){}createBufferSource(){return {connect(){},stop(){},start(...a){starts.push(a);}};}}
 const win={FinalReviewState:F,AudioContext,confirm:()=>true,scrollTo:({left,top})=>{viewport.scrollX=left;viewport.scrollY=top;}};
 Object.defineProperties(win,{scrollX:{get:()=>viewport.scrollX},scrollY:{get:()=>viewport.scrollY}});
 vm.runInNewContext(fs.readFileSync('tools/final_test_review_gui/app.js','utf8'),{window:win,document:{getElementById:id=>ids[id],createElement:tag=>new Node(tag)},localStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v)},structuredClone,Uint8Array,atob:s=>Buffer.from(s,'base64').toString('binary'),URL:{createObjectURL:()=> 'blob:test',revokeObjectURL(){}},Blob:class{constructor(parts){exports.push(JSON.parse(parts[0]));}},setTimeout(){}});
 return {ids,nodes:id=>flatten(ids[id]),exports,starts,storage,viewport};
}
test('DOM preserves six independent drafts, scroll, edge playback and export',async()=>{
 const g=mount();g.ids.reviewer.value='human';g.ids.reviewer.events.input();
 const pick=(id,text)=>g.nodes('catalog').find(n=>n.dataset.noteId===id).children.find(n=>n.tag==='button'&&n.textContent===text).click();
 pick('n0','设为首字');pick('n3','设为末字');
 for(const side of ['start','end'])g.nodes(side+'-edge').find(n=>n.textContent==='本字全部完整可听').click();
 g.ids.status.value='confirmed';g.ids.status.events.change();g.ids.affirm.checked=true;g.ids.affirm.events.change();
 await g.nodes('original').find(n=>n.textContent.includes('完整18秒')).click();await g.nodes('start-edge').find(n=>n.textContent.includes('只听切点后')).click();await g.nodes('end-edge').find(n=>n.textContent.includes('只听切点前')).click();
 assert.deepEqual(g.starts,[[0,0,18],[0,1.25,1.25],[0,0,1.25]]);
 for(let i=2;i<=6;i++){g.nodes('windows').find(n=>n.tag==='button'&&n.textContent.startsWith('T0'+i)).click();g.ids.status.value='pending';g.ids.status.events.change();}
 g.nodes('windows').find(n=>n.tag==='button'&&n.textContent.startsWith('T01')).click();assert.equal(g.ids.status.value,'confirmed');assert.equal(g.ids.affirm.checked,true);assert.equal(g.viewport.scrollY,750);
 g.ids.export.click();assert.equal(g.exports.length,1);assert.equal(g.exports[0].answers[0].edges.start.note_id,'n0');
 const restored=mount(g.storage);restored.ids.export.click();assert.deepEqual(restored.exports[0].answers,g.exports[0].answers);
 assert.doesNotMatch(fs.readFileSync('tools/final_test_review_gui/app.js','utf8'),/fetch\(|innerHTML|autoplay/);
});

function confirmedDraft(packet){const d=pending(packet),x=d.answers.T01;x.status='confirmed';x.sequence_confirmed=true;x.comment='原话不变';for(const [side,note_id] of [['start','n0'],['end','n3']])x.edges[side]={note_id,phones:[{phone_index:0,edge_status:'partial'},{phone_index:1,edge_status:'full'}]};return d;}

test('unresolved interior cannot block export or alter confirmed human answers',()=>{
 const p=fixture();p.songs[0].catalog[1].mapping_status='unresolved';p.songs[0].catalog[1].phones=[];
 const d=confirmedDraft(p),before=structuredClone(d),a=F.exportAll(p,d,'human');
 assert.equal(a.answers[0].status,'confirmed');assert.equal(a.answers[0].comment,'原话不变');assert.equal(a.answers[0].edges.start.phones[0].edge_status,'partial');
 assert.equal(F.unresolvedNotes(p,d.answers.T01)[0].note_id,'n1');assert.deepEqual(d,before);
 assert.deepEqual(F.restore(p,a).answers.T01.edges,d.answers.T01.edges);
});

test('unknown edge exports location and whole-word audibility without inventing IPA',()=>{
 const p=fixture();p.songs[0].catalog[0].mapping_status='unresolved';p.songs[0].catalog[0].phones=[];
 const d=confirmedDraft(p);d.answers.T01.edges.start={note_id:'n0',phones:[],whole_note_audibility:'partial'};
 const a=F.exportAll(p,d,'human');assert.equal(a.answers[0].status,'located');assert.equal(a.unconfirmed_drafts.T01.status,'confirmed');
 assert.equal(a.answers[0].edges.start.whole_note_audibility,'partial');assert.deepEqual(a.answers[0].edges.start.phones,[]);
 const restored=F.restore(p,a);assert.deepEqual(restored.answers.T01.edges,d.answers.T01.edges);
 assert.equal(restored.answers.T01.comment,'原话不变');
});

test('DOM exports an old saved confirmed draft with unresolved mapping after refresh',()=>{
 const p=fixture();p.songs[0].catalog[1].mapping_status='unresolved';p.songs[0].catalog[1].phones=[];
 const all=confirmedDraft(p),storage=new Map([['final-test-review:'+p.packet_sha256,JSON.stringify({packet_sha256:p.packet_sha256,all,current:'T01',reviewer:'human'})]]);
 const g=mount(storage,p);assert.match(g.ids['mapping-status'].textContent,/可以照常导出/);assert.equal(g.ids.status.value,'confirmed');
 g.ids.export.click();assert.equal(g.exports.length,1);assert.equal(g.exports[0].answers[0].status,'confirmed');assert.match(g.ids.notice.textContent,/已保留全部作答/);
 assert.equal(g.viewport.scrollY,750);assert.deepEqual(g.exports[0].unconfirmed_drafts.T01.edges,all.answers.T01.edges);
});
