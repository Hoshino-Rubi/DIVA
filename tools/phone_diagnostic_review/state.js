(function(root,factory){if(typeof module==='object'&&module.exports)module.exports=factory();else root.PhoneDiagnosticReview=factory();})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const decisions=['pending','a','b','neither','uncertain','wrong_location'];
function validate(packet,answers){
 if(!answers||answers.schema_version!=='phone_diagnostic_review_answers_v1'||answers.packet_sha256!==packet.packet_sha256)throw Error('回答与本次页面不匹配。');
 if(!Array.isArray(answers.responses)||answers.responses.length!==packet.items.length)throw Error('回答条目不完整；未答项目也应保留 pending。');
 const seen=new Set();
 for(const row of answers.responses){
  if(!row||seen.has(row.review_id)||!packet.items.some(x=>x.review_id===row.review_id))throw Error('重复或未知的题号。');
  seen.add(row.review_id);
  if(!decisions.includes(row.decision)||typeof row.statement!=='string')throw Error('不支持的回答格式。');
 }
 return answers;
}
function blank(packet){return {schema_version:'phone_diagnostic_review_answers_v1',packet_sha256:packet.packet_sha256,responses:packet.items.map(x=>({review_id:x.review_id,decision:'pending',statement:''})),scope:'diagnostic_only_no_reference_or_training_update'};}
function setAnswer(packet,answers,id,decision,statement){
 const out=JSON.parse(JSON.stringify(validate(packet,answers)));const row=out.responses.find(x=>x.review_id===id);
 if(!row)throw Error('未知题号。');row.decision=decision;row.statement=statement;return validate(packet,out);
}
function exportAnswers(packet,answers){const out=JSON.parse(JSON.stringify(validate(packet,answers)));out.exported_at_utc=new Date().toISOString();out.scope='diagnostic_only_no_reference_or_training_update';return out;}
return {validate,blank,setAnswer,exportAnswers};
});
