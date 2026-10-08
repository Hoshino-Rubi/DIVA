(function(root){
 'use strict';
 const requireValue=(v,m)=>{if(!v)throw Error(m);};
 function empty(packet,windowId){return {window_id:windowId||packet.windows[0].window_id,status:'',comment:'',sequence_confirmed:false,edges:{start:{note_id:null,phones:[]},end:{note_id:null,phones:[]}}};}
 function setNote(draft,side,id){const out=structuredClone(draft);out.edges[side]={note_id:id,phones:[]};out.sequence_confirmed=false;return out;}
 function selected(packet,draft){const lo=packet.catalog.findIndex(n=>n.note_id===draft.edges.start.note_id),hi=packet.catalog.findIndex(n=>n.note_id===draft.edges.end.note_id);return lo>=0&&hi>=lo?packet.catalog.slice(lo,hi+1):[];}
 function validate(packet,payload){
  requireValue(payload?.schema_version==='free_location_review_answers_v1'&&payload.packet_sha256===packet.packet_sha256,'答案不属于此页面');
  requireValue(typeof payload.reviewer==='string'&&payload.reviewer.trim()&&payload.reviewer.length<=120,'请填写复核人');
  requireValue(payload.training_started===false&&!('answers' in payload),'只可提交一个训练窗口');
  const a=payload.answer;requireValue(a&&['confirmed','pending','unavailable'].includes(a.status),'请选择本次结论');
  requireValue(packet.windows.some(w=>w.window_id===a.window_id),'原唱片段无效');
  requireValue(typeof a.comment==='string'&&a.comment.length<=4000,'补充说明过长');
  if(a.status!=='confirmed'){requireValue(a.sequence_confirmed===false&&a.edges&&Object.keys(a.edges).length===0,'待定不能携带参考确认');return structuredClone(payload);}
  requireValue(a.sequence_confirmed===true&&a.edges&&Object.keys(a.edges).sort().join(',')==='end,start','请明确确认连续段落与首尾');
  for(const side of ['start','end']){
   const e=a.edges[side],n=packet.catalog.find(n=>n.note_id===e.note_id);
   requireValue(e.review_id===a.window_id+(side==='start'?'-S':'-E')&&e.status==='confirmed'&&e.observation_schema==='phone_audibility_v2','边缘记录格式无效');
   requireValue(n&&n.mapping_status==='resolved_proxy'&&n.phones.length,'请选择映射已解决的首尾字');
   requireValue(e.audible_identity_confirmed===true&&typeof e.statement==='string'&&e.statement.trim()&&e.statement.length<=4000,'需要明确边缘确认');
   requireValue(Array.isArray(e.phones)&&e.phones.length===n.phones.length,'首尾每个音素都需要选择状态');
   e.phones.forEach((p,i)=>requireValue(p.phone_index===i&&['full','partial','absent','uncertain'].includes(p.edge_status),'无效音素记录'));
   requireValue(e.phones.some(p=>p.edge_status==='full'||p.edge_status==='partial'),'这个字完全听不到，请选择实际可听的边缘字');
  }
  const notes=selected(packet,a);requireValue(notes.length,'首字不能在末字之后');
  // This final-test form records human answers. Unknown interior mappings
  // remain in the catalog and are excluded from scoring by the Python assembler.
  if(a.edges.start.note_id===a.edges.end.note_id)requireValue(JSON.stringify(a.edges.start.phones)===JSON.stringify(a.edges.end.phones),'同一个字的两端记录必须一致');
  return structuredClone(payload);
 }
 function exportAnswer(packet,draft,reviewer){
  const answer={window_id:draft.window_id,status:draft.status,comment:draft.comment,sequence_confirmed:false,edges:{}};
  if(draft.status==='confirmed'){
   answer.sequence_confirmed=draft.sequence_confirmed;
   for(const side of ['start','end'])answer.edges[side]={...structuredClone(draft.edges[side]),review_id:draft.window_id+(side==='start'?'-S':'-E'),status:'confirmed',observation_schema:'phone_audibility_v2',audible_identity_confirmed:draft.sequence_confirmed,statement:'我根据原唱听觉，在完整参考中手动选定'+(side==='start'?'首字':'末字')+'，逐音素记录可听状态；中间为连续参考，没有跳过发音。'};
  }
  return validate(packet,{schema_version:'free_location_review_answers_v1',packet_sha256:packet.packet_sha256,reviewer,exported_at:new Date().toISOString(),answer,unconfirmed_draft:draft.status==='confirmed'?null:structuredClone(draft),training_started:false});
 }
 function fromPayload(packet,payload){
  validate(packet,payload);const a=payload.answer,d=empty(packet,a.window_id);d.status=a.status;d.comment=a.comment;
  if(a.status==='confirmed'){d.sequence_confirmed=true;for(const side of ['start','end'])d.edges[side]={note_id:a.edges[side].note_id,phones:structuredClone(a.edges[side].phones)};}
  else if(payload.unconfirmed_draft){
   const raw=payload.unconfirmed_draft;requireValue(raw.window_id===a.window_id,'待定草稿属于其他片段');
   for(const side of ['start','end']){
    const e=raw.edges?.[side],n=packet.catalog.find(n=>n.note_id===e?.note_id);
    requireValue(e&&Array.isArray(e.phones)&&(n||e.note_id===null),'无效待定草稿');
    const ids=new Set();for(const p of e.phones){requireValue(n&&Number.isInteger(p.phone_index)&&p.phone_index>=0&&p.phone_index<n.phones.length&&!ids.has(p.phone_index)&&['full','partial','absent','uncertain'].includes(p.edge_status),'无效待定音素');ids.add(p.phone_index);}
    d.edges[side]=structuredClone(e);
   }
  }
  return d;
 }
 function searchIndices(packet,query){
  const clean=s=>s.normalize('NFKC').replace(/[\s·，,。！？、]/gu,'');const q=clean(query);
  if(!q)return packet.catalog.map((_,i)=>i);
  let text='',positions=[];packet.catalog.forEach((n,i)=>{const t=clean(n.lyric);text+=t;positions.push(...Array(t.length).fill(i));});
  const ids=new Set();let start=text.indexOf(q);while(start>=0){const lo=positions[start],hi=positions[start+q.length-1];for(let i=Math.max(0,lo-5);i<=Math.min(packet.catalog.length-1,hi+5);i++)ids.add(i);start=text.indexOf(q,start+1);}return [...ids].sort((a,b)=>a-b);
 }
 const api={empty,setNote,selected,validate,exportAnswer,fromPayload,searchIndices};if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.FreeLocationState=api;
})(typeof globalThis!=='undefined'?globalThis:this);

(function(root){
 'use strict';
 const base=typeof module!=='undefined'&&module.exports?module.exports:root.FreeLocationState;
 const need=(v,m)=>{if(!v)throw Error(m);};
 function packetFor(packet,id){const item=packet.items.find(i=>i.window_id===id),song=packet.songs.find(s=>s.song_id===item?.song_id);need(item&&song,'未知片段');return {...packet,...song,windows:[item]};}
 function blank(packet){return {answers:Object.fromEntries(packet.items.map(i=>[i.window_id,base.empty(packetFor(packet,i.window_id),i.window_id)])),mapping_responses:packet.mapping_questions.map(q=>({token:q.token,status:'pending',description:''}))};}
 function unresolvedNotes(packet,draft){return base.selected(packetFor(packet,draft.window_id),draft).filter(n=>n.mapping_status!=='resolved_proxy'||!n.phones.length);}
 function unknownEdge(packet,draft){const single=packetFor(packet,draft.window_id);return ['start','end'].some(side=>{const n=single.catalog.find(n=>n.note_id===draft.edges?.[side]?.note_id);return n&&(n.mapping_status!=='resolved_proxy'||!n.phones.length);});}
 function exportAll(packet,drafts,reviewer){
  need(typeof reviewer==='string'&&reviewer.trim()&&reviewer.length<=120,'请填写复核人');
  const answers=packet.items.map(item=>{
   const d=drafts.answers[item.window_id],single=packetFor(packet,item.window_id);
   if(d.status==='located'||(d.status==='confirmed'&&unknownEdge(packet,d))){
    need(d.sequence_confirmed&&base.selected(single,d).length,'定位确认需要有效首尾及连续范围确认');
    return {window_id:d.window_id,status:'located',comment:d.comment,sequence_confirmed:true,edges:Object.fromEntries(['start','end'].map(side=>[side,{note_id:d.edges[side].note_id,phones:[],whole_note_audibility:d.edges[side].whole_note_audibility||'uncertain'}]))};
   }
   return base.exportAnswer(single,d,reviewer).answer;
  });
  const out={schema_version:'frozen_final_test_review_answers_v1',packet_sha256:packet.packet_sha256,reviewer,exported_at:new Date().toISOString(),answers,mapping_responses:structuredClone(drafts.mapping_responses),unconfirmed_drafts:structuredClone(drafts.answers),training_started:false,test_predictions_opened:false};
  validate(packet,out);return out;
 }
 function validate(packet,payload){
  need(payload.schema_version==='frozen_final_test_review_answers_v1'&&payload.packet_sha256===packet.packet_sha256,'答案属于其他最终审计');
  need(typeof payload.reviewer==='string'&&payload.reviewer.trim()&&payload.reviewer.length<=120,'请填写复核人');
  need(payload.training_started===false&&payload.test_predictions_opened===false,'答案必须保持盲审');
  need(Array.isArray(payload.answers)&&payload.answers.length===6&&new Set(payload.answers.map(a=>a.window_id)).size===6,'需要六段记录，不能重复');
  for(const a of payload.answers){
   const single=packetFor(packet,a.window_id);
   if(a.status==='located'){
    need(a.sequence_confirmed&&a.edges&&Object.keys(a.edges).length===2&&base.selected(single,a).length,'无效定位范围');
    for(const side of ['start','end'])need(a.edges[side].phones.length===0&&['full','partial','uncertain'].includes(a.edges[side].whole_note_audibility||'uncertain'),'定位答案不自动确认IPA');
   }else base.validate(single,{schema_version:'free_location_review_answers_v1',packet_sha256:packet.packet_sha256,reviewer:payload.reviewer,answer:a,training_started:false});
  }
  need(payload.mapping_responses?.length===packet.mapping_questions.length&&new Set(payload.mapping_responses.map(m=>m.token)).size===5,'需要五项符号说明');
  for(const m of payload.mapping_responses)need(packet.mapping_questions.some(q=>q.token===m.token)&&['described','pending'].includes(m.status)&&typeof m.description==='string'&&m.description.length<=4000&&(m.status!=='described'||m.description.trim()),'符号说明不完整');
  return payload;
 }
 function restore(packet,payload){
  validate(packet,payload);const out=blank(packet);
  for(const a of payload.answers){
   if(a.status==='located'){
    const raw=payload.unconfirmed_drafts?.[a.window_id];
    const d=raw?base.fromPayload(packetFor(packet,a.window_id),{schema_version:'free_location_review_answers_v1',packet_sha256:packet.packet_sha256,reviewer:payload.reviewer,answer:{window_id:a.window_id,status:'pending',comment:a.comment,sequence_confirmed:false,edges:{}},training_started:false,unconfirmed_draft:raw}):base.empty(packetFor(packet,a.window_id),a.window_id);
    d.status='located';d.comment=a.comment;d.sequence_confirmed=true;if(!raw)d.edges=structuredClone(a.edges);out.answers[a.window_id]=d;
   }else out.answers[a.window_id]=base.fromPayload(packetFor(packet,a.window_id),{schema_version:'free_location_review_answers_v1',packet_sha256:packet.packet_sha256,reviewer:payload.reviewer,answer:a,training_started:false,unconfirmed_draft:['pending','unavailable'].includes(a.status)?payload.unconfirmed_drafts?.[a.window_id]:null});
  }
  out.mapping_responses=structuredClone(payload.mapping_responses);return out;
 }
 const api={base,packetFor,blank,unresolvedNotes,exportAll,validate,restore};if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.FinalReviewState=api;
})(typeof globalThis!=='undefined'?globalThis:this);
