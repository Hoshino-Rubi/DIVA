(function(root){
 'use strict';
 const SCHEMA='final_test_proxy_mapping_review_answers_v1';
 const need=(v,m)=>{if(!v)throw Error(m);};
 function blank(packet){return packet.rules.map(r=>({token:r.token,status:'',option_id:null,ipa:[],comment:''}));}
 function choose(packet,rows,token,choice){
  const copy=structuredClone(rows),d=copy.find(x=>x.token===token),r=packet.rules.find(x=>x.token===token);
  need(d&&r,'未知符号');d.option_id=null;d.ipa=[];
  if(choice===''||choice==='defer'||choice==='custom_proxy'){d.status=choice;return copy;}
  const o=r.options.find(x=>x.id===choice);need(o,'无效提案');d.status='approve_proxy';d.option_id=o.id;d.ipa=structuredClone(o.ipa);return copy;
 }
 function acceptFour(packet,rows){let copy=structuredClone(rows);for(const r of packet.rules){if(r.token!=='UN'&&r.bulk_option_id&&copy.find(x=>x.token===r.token).status==='')copy=choose(packet,copy,r.token,r.bulk_option_id);}return copy;}
 function validate(packet,payload){
  need(payload?.schema_version===SCHEMA&&payload.packet_sha256===packet.packet_sha256,'答案不属于本页');
  need(typeof payload.reviewer==='string'&&payload.reviewer.trim()&&payload.reviewer.trim().length<=120,'请填写复核人');
  need(payload.training_started===false&&payload.test_predictions_opened===false,'答案必须保持盲审');
  need(Array.isArray(payload.decisions)&&payload.decisions.length===5&&new Set(payload.decisions.map(d=>d.token)).size===5,'需要五个不同符号的记录');
  for(const d of payload.decisions){
   const r=packet.rules.find(r=>r.token===d.token);need(r,'未知符号');need(['approve_proxy','custom_proxy','defer'].includes(d.status),'无效结论');need(typeof d.comment==='string'&&d.comment.length<=4000,'原话过长');
   if(d.status==='defer')need(d.option_id===null&&Array.isArray(d.ipa)&&d.ipa.length===0,'待定不能附带已确认 IPA');
   if(d.status==='approve_proxy'){const o=r.options.find(o=>o.id===d.option_id);need(o&&JSON.stringify(d.ipa)===JSON.stringify(o.ipa),'提案与所选 IPA 不一致');}
   if(d.status==='custom_proxy')need(d.option_id===null&&Array.isArray(d.ipa)&&d.ipa.length>0&&d.ipa.length<=8&&d.ipa.every(p=>packet.allowed_ipa.includes(p)),'自定义 IPA 请用支持的音素，以空格分隔；不能包含 blank');
  }
  const approved=payload.decisions.some(d=>d.status!=='defer');
  need(!approved||payload.consent===packet.consent_scope,'请勾选代理评测约定；不是要求你确认精确原唱 IPA');
  need(payload.authorize_frozen_proxy_comparison===approved,'评测授权与结论不一致');return structuredClone(payload);
 }
 function restoreDraft(packet,cache){
  need(cache?.packet_sha256===packet.packet_sha256&&Array.isArray(cache.rows)&&cache.rows.length===5&&new Set(cache.rows.map(d=>d.token)).size===5,'草稿不属于本页');
  for(const d of cache.rows){
   const r=packet.rules.find(r=>r.token===d.token);need(r&&['','defer','approve_proxy','custom_proxy'].includes(d.status),'无效草稿条目');
   need(typeof d.comment==='string'&&Array.isArray(d.ipa)&&d.ipa.every(p=>typeof p==='string'),'无效草稿记录');
   if(d.status==='approve_proxy'){const o=r.options.find(o=>o.id===d.option_id);need(o&&JSON.stringify(o.ipa)===JSON.stringify(d.ipa),'草稿提案已变化');}
   if(d.status===''||d.status==='defer')need(d.option_id===null&&d.ipa.length===0,'无效待定草稿');
  }
  return structuredClone(cache.rows);
 }
 function exportAll(packet,rows,reviewer,consent){
  const decisions=rows.map(d=>d.status===''?{...structuredClone(d),status:'defer',option_id:null,ipa:[]}:structuredClone(d));
  const approved=decisions.some(d=>d.status!=='defer');
  return validate(packet,{schema_version:SCHEMA,packet_sha256:packet.packet_sha256,reviewer,exported_at:new Date().toISOString(),decisions,
   consent:consent?packet.consent_scope:null,authorize_frozen_proxy_comparison:approved,training_started:false,test_predictions_opened:false});
 }
 const api={blank,choose,acceptFour,validate,restoreDraft,exportAll};if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.FinalMappingState=api;
})(typeof globalThis!=='undefined'?globalThis:this);
