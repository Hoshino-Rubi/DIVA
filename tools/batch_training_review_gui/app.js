(function(){
 'use strict';
 const S=window.BatchReviewState,$=id=>document.getElementById(id);
 const packet=JSON.parse($('packet').textContent),audioData=JSON.parse($('audio-data').textContent);
 const key='batch-training-review:'+packet.packet_sha256;
 const names={'freedom-flower':'自由花','queen-road-east':'皇后大道東','half-moon-night-song':'月半小夜曲','promise':'约定','like-you':'喜欢你'};
 let current=0,drafts={},answers={},context=null,source=null,epoch=0;const buffers=new Map();
 const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
 const button=(text,fn,cls)=>{const b=el('button',text,cls);b.type='button';b.addEventListener('click',fn);return b;};
 const notice=(text,error=false)=>{$('notice').textContent=text;$('notice').className=error?'error':'';};
 const item=()=>packet.items[current];
 const draft=()=>drafts[item().review_id]||=(S.empty(item()));
 function save(loud=false){try{localStorage.setItem(key,JSON.stringify({packet_sha256:packet.packet_sha256,current,drafts,answers,reviewer:$('reviewer').value}));if(loud)notice('草稿已保存在本机。');}catch(_){notice('本机保存不可用，请提交并导出答案备份。',true);}}
 function change(fn){fn(draft());draft().sequence_confirmed=false;delete answers[item().review_id];save();navigation();const box=$('affirm');if(box)box.checked=false;}
 function stop(){epoch++;if(source){try{source.stop();}catch(_){}source=null;}for(const a of document.querySelectorAll('audio'))a.pause();}
 async function playEdge(media,side,full=false){
  stop();const token=epoch;
  try{context||=new(window.AudioContext||window.webkitAudioContext)();
   if(!buffers.has(media.audio_file)){const bytes=Uint8Array.from(atob(audioData[media.audio_file]),x=>x.charCodeAt(0));buffers.set(media.audio_file,context.decodeAudioData(bytes.buffer));}
   const buffer=await buffers.get(media.audio_file);await context.resume();if(token!==epoch)return;
   if(Math.abs(buffer.duration-2.5)>.002)throw Error('音频时长与正式切点不一致');
   source=context.createBufferSource();source.buffer=buffer;source.connect(context.destination);
   source.start(0,full?0:side==='start'?1.25:0,full?2.5:1.25);
   notice(full?'播放完整上下文，红色中线才是正式切点。':'播放正式片段内侧：'+(side==='start'?'中线右侧':'中线左侧'));
  }catch(e){notice('播放失败：'+e.message,true);}
 }
 function mediaPanel(media,title){const box=el('div');box.append(el('h3',title));const img=el('img');img.className='wave';img.src='media/'+media.waveform_file;img.alt=title+'波形';const audio=el('audio');audio.controls=true;audio.preload='none';audio.src='media/'+media.audio_file;audio.addEventListener('play',()=>{epoch++;if(source){try{source.stop();}catch(_){}source=null;}for(const other of document.querySelectorAll('audio'))if(other!==audio)other.pause();});box.append(img,audio);return box;}
 function navigation(){const nav=$('navigation');nav.replaceChildren();packet.items.forEach((r,i)=>nav.append(button(r.review_id+' · '+names[r.song_id]+(answers[r.review_id]?' · 已提交':drafts[r.review_id]?' · 草稿':''),()=>{stop();save();current=i;render();},i===current?'active':'')));$('progress').textContent='已提交 '+Object.keys(answers).length+' / '+packet.items.length;}
 function preserve(fn){const x=window.scrollX||0,y=window.scrollY||0;fn();if(window.scrollTo)window.scrollTo({left:x,top:y,behavior:'instant'});}
 function renderEdge(side,choice,host){
  host.replaceChildren();const r=item(),d=draft(),ed=d.edges[side];host.className='edge';
  const head=el('div',undefined,'edge-heading');head.append(el('h3',side==='start'?'首字与音素':'末字与音素'),el('span','原唱 '+r[side+'_sec'].toFixed(4)+' 秒 · 红线 1.25 秒','muted'));host.append(head);
  const wrap=el('div',undefined,'cut '+side),img=el('img');img.src='media/'+r.edge_media[side].waveform_file;img.className='wave';img.alt='红线为正式切点，绿色为片段内侧';wrap.append(img);host.append(wrap);
  const controls=el('div',undefined,'actions');controls.append(button('▶ '+(side==='start'?'听右半：片段内':'听左半：片段内'),()=>playEdge(r.edge_media[side],side)),button('▶ 完整上下文',()=>playEdge(r.edge_media[side],side,true)),button('■ 停止',stop));host.append(controls);
  if(!choice){host.append(el('p','尚未选定对应段落。','muted'));return;}
  if(choice.edge_score_gaps?.includes(side))host.append(el('p','此侧旧参考切点附近没有音符。请以十秒原唱中第一个／最后一个实际可听字为准；候选仅供定位，不代表已确认。','warning'));
  const select=el('select');select.setAttribute('aria-label',side==='start'?'首字':'末字');let o=el('option','请选择实际首字 / 末字');o.value='';select.append(o);
  const nearby=new Set(choice.edge_notes[side].map(n=>n.note_id));
  for(const n of choice.catalog){o=el('option',(nearby.has(n.note_id)?'邻近 · ':'')+n.lyric+' /'+n.phones.join(' ')+'/ · '+n.note_id+(n.mapping_status!=='resolved_proxy'?' · 映射未解决':''));o.value=n.note_id;o.disabled=n.mapping_status!=='resolved_proxy';select.append(o);}
  select.value=ed.note_id||'';select.addEventListener('change',()=>{change(x=>{x.edges[side]={note_id:select.value||null,phones:[]};});preserve(()=>renderEdge(side,choice,host));updateRange(choice);});host.append(select);
  const note=choice.catalog.find(n=>n.note_id===ed.note_id);if(!note)return;
  host.append(button('本字全部完整可听',()=>{change(x=>{x.edges[side].phones=note.phones.map((_,i)=>({phone_index:i,edge_status:'full'}));});preserve(()=>renderEdge(side,choice,host));}));
  note.phones.forEach((phone,index)=>{const row=el('div',undefined,'phone-row');row.append(el('span','/'+phone+'/ · '+index,'ipa'));const opts=el('div',undefined,'phone-options');const state=ed.phones.find(p=>p.phone_index===index)?.edge_status||'';
   [['','未判断'],['full','完整可听'],['partial','部分可听'],['absent','完全听不到'],['uncertain','不确定']].forEach(([value,label])=>{
    const b=button(label,()=>{change(x=>{const e=x.edges[side];e.phones=e.phones.filter(p=>p.phone_index!==index);if(value)e.phones.push({phone_index:index,edge_status:value});e.phones.sort((a,b)=>a.phone_index-b.phone_index);});for(const sibling of opts.children)sibling.className=sibling===b?'active':'';},state===value?'active':'');opts.append(b);
   });row.append(opts);host.append(row);
  });
 }
 function updateRange(choice){const target=$('sequence');if(!target)return;const d=draft();const lo=choice.catalog.findIndex(n=>n.note_id===d.edges.start.note_id),hi=choice.catalog.findIndex(n=>n.note_id===d.edges.end.note_id);const notes=lo>=0&&hi>=lo?choice.catalog.slice(lo,hi+1):[];const unknown=notes.filter(n=>n.mapping_status!=='resolved_proxy');target.textContent=notes.length?notes.map(n=>n.lyric).join(' · ')+(unknown.length?'（范围内仍有未解决映射：'+unknown.map(n=>n.lyric).join('、')+'；该段不能通过参考验收）':''):'首尾范围尚未确定';}
 function render(){preserve(renderContents);}
 function renderContents(){
  navigation();const r=item(),d=draft(),root=$('review');root.replaceChildren();
  const intro=el('div',undefined,'band');intro.append(el('h2',r.review_id+' · '+names[r.song_id]),el('p',r.start_sec.toFixed(3)+'–'+r.end_sec.toFixed(3)+' 秒 · '+(r.kind==='retained'?'已有对应，补齐边缘':'新候选，对应尚未确认')));
  if(r.review_focus)intro.append(el('p',r.review_focus,'warning'));
  if(r.previous_audit){const details=el('details');details.open=!r.previous_answer;details.append(el('summary','既有审核记录'));details.append(el('pre',JSON.stringify(r.previous_audit.final_audit_response||r.previous_audit,null,2)));intro.append(details);}
  intro.append(mediaPanel(r.original,'原唱 · 十秒正式窗口'));root.append(intro);
  const choices=el('div',undefined,'band');choices.append(el('h3',r.correspondence_preconfirmed?'已确认对应 · 合成辅助':'段落对应'));
  const grid=el('div',undefined,'alternatives');
  for(const c of r.alternatives){const box=el('div',undefined,'alternative'),label=el('label'),radio=el('input');radio.type='radio';radio.name='choice';radio.value=c.choice_id;radio.checked=d.choice===c.choice_id;radio.disabled=r.correspondence_preconfirmed;radio.addEventListener('change',()=>{stop();drafts[r.review_id]=S.setChoice(r,draft(),c.choice_id);delete answers[r.review_id];save();render();});label.append(radio,document.createTextNode(c.choice_id+' · '+c.start_sec.toFixed(2)+'–'+c.end_sec.toFixed(2)+' 秒'));box.append(label,mediaPanel(c.preview,'合成备选 '+c.choice_id));const details=el('details');details.append(el('summary','附近校准参考'),el('p',c.notes.map(n=>n.lyric).join(' · ')));box.append(details);if(!c.mapping_complete)box.append(el('p','包含未解决映射','warning'));grid.append(box);}choices.append(grid);root.append(choices);
  const choice=r.alternatives.find(c=>c.choice_id===d.choice);for(const side of ['start','end']){const host=el('div');root.append(host);renderEdge(side,choice,host);}
  const range=el('div',undefined,'band');range.append(el('h3','连续参考范围'));const sequence=el('p',undefined,'sequence');sequence.id='sequence';range.append(sequence);root.append(range);if(choice)updateRange(choice);else sequence.textContent='尚未选择对应';
  const statusBox=el('label','本段结论','status'),status=el('select');status.setAttribute('aria-label','本段结论');[['','请选择'],['confirmed','提交对应与两端记录'],['uncertain','待定，暂不确认'],['rejected','对应错误 / 均不对应']].forEach(([v,t])=>{const op=el('option',t);op.value=v;status.append(op);});status.value=d.status;status.addEventListener('change',()=>{change(x=>{x.status=status.value;});});statusBox.append(status);root.append(statusBox);
  const affirm=el('label',undefined,'affirmation'),box=el('input');box.type='checkbox';box.id='affirm';box.checked=d.sequence_confirmed;box.addEventListener('change',()=>{draft().sequence_confirmed=box.checked;delete answers[r.review_id];save();navigation();});affirm.append(box,document.createTextNode('我确认此段对应及所选首尾身份；中间为连续参考，没有跳过片段内的字。音素可听状态来自听觉，不按歌词猜测。'));root.append(affirm);
  const comment=el('textarea');comment.value=d.comment;comment.placeholder='补充原话 / 无法确认的原因（可选）';comment.maxLength=4000;comment.addEventListener('input',()=>{draft().comment=comment.value;delete answers[r.review_id];save();navigation();});root.append(comment);
  const actions=el('div',undefined,'actions');actions.append(button('提交本段，下一段',()=>{try{const a=S.commit(packet,r,draft(),$('reviewer').value.trim());answers[r.review_id]=a;stop();if(current<packet.items.length-1)current++;save();render();notice('本段已提交。已提交 '+Object.keys(answers).length+' / '+packet.items.length+'。');}catch(e){notice(e.message,true);}},'primary'),button('保存草稿',()=>save(true)),button('下一段（不提交）',()=>{stop();save();current=(current+1)%packet.items.length;render();}));root.append(actions);
 }
 $('save').addEventListener('click',()=>save(true));$('reviewer').addEventListener('input',()=>save());
 $('export').addEventListener('click',()=>{try{const payload=S.exportAnswers(packet,Object.values(answers),$('reviewer').value.trim());if(!payload.answers.length)throw Error('尚无已提交答案');const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));const a=el('a');a.href=url;a.download='batch-training-review-answers-'+packet.packet_sha256.slice(0,12)+'.json';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),30000);notice('已导出 '+payload.answers.length+' 段；未提交草稿不计入。');}catch(e){notice(e.message,true);}});
 $('import').addEventListener('change',async event=>{try{const file=event.target.files[0];if(!file)return;const payload=S.validate(packet,JSON.parse(await file.text()));if(!window.confirm('导入 '+payload.answers.length+' 段答案？同编号本机记录将被替换。'))return;stop();$('reviewer').value=payload.reviewer;for(const a of payload.answers){answers[a.review_id]=a;drafts[a.review_id]=S.draftFromAnswer(packet.items.find(i=>i.review_id===a.review_id),a);}save();render();notice('导入成功。');}catch(e){notice('导入失败：'+e.message,true);}finally{event.target.value='';}});
 try{const saved=JSON.parse(localStorage.getItem(key)||'null');if(saved&&saved.packet_sha256===packet.packet_sha256){$('reviewer').value=saved.reviewer||'';const list=Object.values(saved.answers||{});if(list.length)S.exportAnswers(packet,list,saved.reviewer);answers=saved.answers||{};for(const r of packet.items){const d=saved.drafts?.[r.review_id];if(d&&typeof d.comment==='string'&&d.edges?.start&&d.edges?.end&&Array.isArray(d.edges.start.phones)&&Array.isArray(d.edges.end.phones))drafts[r.review_id]=d;}current=Number.isInteger(saved.current)&&saved.current>=0&&saved.current<packet.items.length?saved.current:0;notice('已恢复本机进度。');}}catch(_){drafts={};answers={};notice('旧本机进度未通过检查；可使用同批答案文件恢复。',true);}
 render();
})();
