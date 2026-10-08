(function(){
 'use strict';
 const $=id=>document.getElementById(id),S=window.FinalMappingState,P=JSON.parse($('packet').textContent),key='final-mapping-review:'+P.packet_sha256;
 let rows=S.blank(P);const views=new Map();
 $('reviewer').value=P.previous_reviewer;
 const tell=m=>{$('notice').textContent=m;};
 function save(){try{localStorage.setItem(key,JSON.stringify({packet_sha256:P.packet_sha256,rows,reviewer:$('reviewer').value,consent:$('consent').checked}));}catch(e){tell('浏览器草稿保存失败；仍可直接导出 JSON。');}}
 try{const raw=JSON.parse(localStorage.getItem(key)||'null');if(raw){if(raw.packet_sha256!==P.packet_sha256)throw Error('草稿属于其他页');rows=S.restoreDraft(P,raw);$('reviewer').value=raw.reviewer;$('consent').checked=raw.consent;}}catch(e){tell('旧草稿未载入：'+e.message);}
 function node(tag,text){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n;}
 function update(){let approved=0;for(const d of rows){const v=views.get(d.token);v.select.value=d.status==='approve_proxy'?d.option_id:d.status;v.custom.hidden=d.status!=='custom_proxy';v.custom.value=d.ipa.join(' ');v.comment.value=d.comment;v.status.textContent=d.status==='approve_proxy'||d.status==='custom_proxy'?'已选择代理规则，导出时保存':'尚未批准映射';v.card.dataset.status=d.status;if(['approve_proxy','custom_proxy'].includes(d.status))approved++;}$('progress').textContent='已选择 '+approved+' / 5 项；其余可作为待定导出。';}
 function render(){
  for(const r of P.rules){
   const card=node('article'),title=node('h2',r.token);card.id='rule-'+r.token;card.dataset.token=r.token;card.append(title,node('p','涉及：'+r.affected.map(a=>a.window_id+'「'+a.lyric+'」').join('、')));
   const quote=node('blockquote',r.human_statement);card.append(node('p','你已提交的说明（无需重写）：'),quote,node('p',r.basis),node('p','声库手册第 '+r.manual.page+' 页示例：'+r.manual.sample));
   const label=node('label','本项转换约定'),select=node('select');select.id='choice-'+r.token;
   for(const [value,text] of [['','请选择；不选也可导出为待定'],...r.options.map(o=>[o.id,o.label+'  /'+o.ipa.join(' ')+'/']),['custom_proxy','我提供其他 IPA 代理规则（可选）'],['defer','先保留原写法，不采用这些近似']]){const option=node('option',text);option.value=value;select.append(option);}label.append(select);card.append(label);
   for(const o of r.options)card.append(node('p','提案 /'+o.ipa.join(' ')+'/：'+o.assumption));
   const custom=node('input');custom.type='text';custom.placeholder='可选：输入 IPA，以空格分隔；例如 ɔ n';custom.setAttribute('aria-label',r.token+' 自定义 IPA');custom.id='custom-'+r.token;custom.hidden=true;
   const comment=node('textarea');comment.rows=2;comment.maxLength=4000;comment.placeholder='可选：只写本次要补充的说明';comment.setAttribute('aria-label',r.token+' 补充说明');comment.id='comment-'+r.token;
   const status=node('p');status.className='status';card.append(custom,comment,status);$('rules').append(card);views.set(r.token,{card,select,custom,comment,status});
   select.addEventListener('change',()=>{rows=S.choose(P,rows,r.token,select.value);update();save();});
   custom.addEventListener('input',()=>{rows.find(d=>d.token===r.token).ipa=custom.value.trim().normalize('NFC').split(/\s+/u).filter(Boolean);save();});
   comment.addEventListener('input',()=>{rows.find(d=>d.token===r.token).comment=comment.value;save();});
  }
  for(const r of P.manual_audit.rows){const tr=node('tr');for(const t of [r.window_id,r.song_name,r.notes.length?('「'+r.notes[0].lyric+'」→「'+r.notes[r.notes.length-1].lyric+'」'):'待定',r.notes.length+' 个音符',r.unresolved_notes.length+' 个待转换'])tr.append(node('td',t));$('audit-rows').append(tr);}
  update();
 }
 render();
 $('bulk').addEventListener('click',()=>{rows=S.acceptFour(P,rows);update();save();tell('已填入尚未作答的四项单一提案。UN 存在不同约定，需单独选择；已有选择不会被覆盖。');});
 $('reviewer').addEventListener('input',save);$('consent').addEventListener('change',save);$('save').addEventListener('click',()=>{save();tell('草稿已保存到本页。');});
 $('export').addEventListener('click',()=>{try{const answer=S.exportAll(P,rows,$('reviewer').value,$('consent').checked);save();const url=URL.createObjectURL(new Blob([JSON.stringify(answer,null,2)],{type:'application/json'}));const a=node('a');a.href=url;a.download='final-mapping-decisions-'+P.packet_sha256.slice(0,12)+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);tell('已导出五项记录；六段定位和先前音素作答已绑定保留。待定不会拦截导出。');}catch(e){tell(e.message);}});
 $('import').addEventListener('change',async()=>{const f=$('import').files[0];if(!f)return;try{const a=S.validate(P,JSON.parse(await f.text()));rows=a.decisions;$('reviewer').value=a.reviewer;$('consent').checked=a.consent===P.consent_scope;update();save();tell('已载入同页答案。');}catch(e){tell('未覆盖现有草稿：'+e.message);}});
})();
