'use strict';
const packet=JSON.parse(document.getElementById('packet').textContent);
const state=PhoneDiagnosticReview;const storageKey='phone-diagnostic-'+packet.packet_sha256;
let answers=state.blank(packet);
const status=document.getElementById('status');
function showCount(){const n=answers.responses.filter(r=>r.decision!=='pending').length;document.getElementById('count').textContent=`已回答 ${n} / ${packet.items.length}；其余项目可保留待定。`;}
function save(){try{localStorage.setItem(storageKey,JSON.stringify(answers));status.textContent='已保存到本机。';}catch(e){status.textContent='本机存储不可用，请及时导出 JSON。';}showCount();}
function fill(){for(const row of answers.responses){const box=document.getElementById(row.review_id);box.querySelectorAll('input[type=radio]').forEach(el=>el.checked=el.value===row.decision);box.querySelector('textarea').value=row.statement;}showCount();}
try{const saved=localStorage.getItem(storageKey);if(saved)answers=state.validate(packet,JSON.parse(saved));}catch(e){status.textContent='旧草稿未加载：'+e.message;}
fill();
document.querySelectorAll('.question').forEach(box=>{
 function update(){const selected=box.querySelector('input[type=radio]:checked');answers=state.setAnswer(packet,answers,box.id,selected?selected.value:'pending',box.querySelector('textarea').value);save();}
 box.querySelectorAll('input[type=radio]').forEach(el=>el.addEventListener('change',update));
 box.querySelector('textarea').addEventListener('input',update);
});
document.getElementById('export').addEventListener('click',()=>{
 try{const value=state.exportAnswers(packet,answers);const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='phone-diagnostic-review-answers-'+packet.packet_sha256.slice(0,12)+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);status.textContent='已导出；未回答、不确定、定位不对均已保留。';}
 catch(e){status.textContent=e.message;}
});
document.getElementById('import').addEventListener('change',async event=>{
 try{const file=event.target.files[0];if(!file)return;const checked=state.validate(packet,JSON.parse(await file.text()));answers=checked;fill();save();}
 catch(e){status.textContent='导入失败：'+e.message;}
 event.target.value='';
});
