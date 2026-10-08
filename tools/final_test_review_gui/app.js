(function(){
 'use strict';
 const $=id=>document.getElementById(id),F=window.FinalReviewState,S=F.base;
 const batch=JSON.parse($('packet').textContent),audio=JSON.parse($('audio-data').textContent),key='final-test-review:'+batch.packet_sha256;
 let all=F.blank(batch),current=batch.items[0].window_id,packet=F.packetFor(batch,current),draft=all.answers[current],context=null,source=null,epoch=0;const buffers=new Map();
 const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
 const button=(text,fn,cls)=>{const b=el('button',text,cls);b.type='button';b.addEventListener('click',fn);return b;};
 const notice=(text,error=false)=>{$('notice').textContent=text;$('notice').className=error?'error':'';};
 const win=()=>packet.windows.find(w=>w.window_id===draft.window_id);
 function save(loud=false){all.answers[current]=draft;try{localStorage.setItem(key,JSON.stringify({packet_sha256:batch.packet_sha256,all,current,reviewer:$('reviewer').value}));if(loud)notice('草稿已保存在本机。');}catch(_){notice('本机保存失败，请导出 JSON 备份。',true);}}
 function changed(){draft.sequence_confirmed=false;$('affirm').checked=false;save();}
 function preserve(fn){const x=window.scrollX||0,y=window.scrollY||0;fn();window.scrollTo?.({left:x,top:y,behavior:'instant'});}
 function stop(){epoch++;if(source){try{source.stop();}catch(_){}source=null;}}
 async function play(media,offset=0,duration=media.duration_sec,label='原唱'){
  stop();const token=epoch;
  try{
   if(!Number.isFinite(offset)||!Number.isFinite(duration)||offset<0||duration<=0||offset+duration>media.duration_sec+.002)throw Error('播放范围超出音频');
   context||=new(window.AudioContext||window.webkitAudioContext)();
   if(!buffers.has(media.audio_file)){const bytes=Uint8Array.from(atob(audio[media.audio_file]),x=>x.charCodeAt(0));buffers.set(media.audio_file,context.decodeAudioData(bytes.buffer));}
   const buffer=await buffers.get(media.audio_file);await context.resume();if(token!==epoch)return;
   if(Math.abs(buffer.duration-media.duration_sec)>.002)throw Error('音频时长校验失败');
   source=context.createBufferSource();source.buffer=buffer;source.connect(context.destination);source.start(0,offset,duration);notice('正在播放：'+label+'（'+duration.toFixed(2)+' 秒）');
  }catch(e){notice('播放失败：'+e.message,true);}
 }
 function playNotes(notes){if(!notes.length)return notice('请先选定首尾字',true);const start=Math.max(0,notes[0].start_sec-.12),end=Math.min(packet.rendered_preview.duration_sec,notes.at(-1).end_sec+.12);const duration=Math.min(end-start,20);play(packet.rendered_preview,start,duration,'合成参考 '+start.toFixed(2)+' 秒起'+(end-start>20?'，先播放前 20 秒':''));}
 function renderWindows(){
  $('windows').replaceChildren();batch.items.forEach(w=>{
   const song=batch.songs.find(s=>s.song_id===w.song_id),d=all.answers[w.window_id],mark=d?.status?(d.sequence_confirmed?'✓ 已记录':'· 已存草稿'):'· 未填写';
   $('windows').append(button(w.window_id+' '+song.song_name+' '+mark,()=>{
    save();stop();current=w.window_id;packet=F.packetFor(batch,current);draft=all.answers[current];
    $('search').value='';$('status').value=draft.status;$('comment').value=draft.comment;$('affirm').checked=draft.sequence_confirmed;
    preserve(()=>{renderWindows();renderCatalog();renderEdges();});save();
   },w.window_id===current?'active':''));
  });
  $('original').replaceChildren(button('▶ 听完整18秒原唱',()=>play(win().original)),button('■ 停止',stop),el('span',packet.song_name+' · '+win().start_sec.toFixed(3)+'–'+win().end_sec.toFixed(3)+'秒 · 六段都可独立保存','hint'));
 }
 function pick(side,note){draft=S.setNote(draft,side,note.note_id);changed();preserve(()=>{renderEdges();paintSelection();});}
 function paintSelection(){for(const n of $('catalog').querySelectorAll('[data-note-id]')){const note=packet.catalog.find(v=>v.note_id===n.dataset.noteId);n.className='note'+(note.mapping_status!=='resolved_proxy'?' unknown':'')+(['start','end'].some(s=>draft.edges[s].note_id===note.note_id)?' picked':'');}}
 function renderCatalog(){
  const indices=S.searchIndices(packet,$('search').value),groups=new Map();
  for(const i of indices){const group=Math.floor(i/10);if(!groups.has(group))groups.set(group,[]);groups.get(group).push(i);}
  const box=$('catalog');box.replaceChildren();$('search-count').textContent=indices.length?'显示 '+indices.length+' / '+packet.catalog.length+' 个字（含匹配处前后文）；每次只选一个首字和一个末字。':'未找到。可减少搜索字数、核对繁体字，或显示整首。';
  for(const ids of groups.values()){
   const notes=ids.map(i=>packet.catalog[i]),row=el('div',undefined,'phrase'),head=el('div',undefined,'phrase-head');
   head.append(button('▶ 听这一行',()=>playNotes(notes)),el('span','参考 '+notes[0].start_sec.toFixed(2)+'–'+notes.at(-1).end_sec.toFixed(2)+' 秒'));
   const cards=el('div',undefined,'notes');
   for(const note of notes){const card=el('div',undefined,'note');card.dataset.noteId=note.note_id;card.append(el('span',note.lyric,'lyric'),el('small',note.note_id),button('设为首字',()=>pick('start',note)),button('设为末字',()=>pick('end',note)));if(note.mapping_status!=='resolved_proxy')card.append(el('small','映射未解决'));cards.append(card);}
   row.append(head,cards);box.append(row);
  }
  paintSelection();save();
 }
 function renderRange(){const notes=S.selected(packet,draft);$('range').textContent=notes.length?notes.map(n=>n.lyric).join(' · '):'尚未选定有效首尾范围';const unresolved=F.unresolvedNotes(batch,draft);$('mapping-status').textContent=unresolved.length?'所选范围有 '+unresolved.length+' 个字的 IPA 映射待解决：'+unresolved.map(n=>n.lyric).join('、')+'。可以照常导出；首尾选择、音素状态及原话都会保留，相关窗口暂不评分。':'已知映射可用于整理参考；导出会保留你的原始作答。';}
 function renderEdge(side){
  const host=$(side+'-edge');host.className='edge';host.replaceChildren();const selected=draft.edges[side],note=packet.catalog.find(n=>n.note_id===selected.note_id),media=win().edges[side];
  host.append(el('h3',(side==='start'?'首字':'末字')+' · '+(note?note.lyric:'未选')));
  host.append(button('▶ '+(side==='start'?'只听切点后：片段内':'只听切点前：片段内'),()=>play(media,side==='start'?1.25:0,1.25,'正式片段内侧')),button('▶ 切点前后 2.5 秒',()=>play(media,0,2.5,'切点上下文，正式切点在 1.25 秒')));
  if(!note)return;
  host.append(el('p',note.note_id+' · /'+note.phones.join(' ')+'/','hint'));
  if(note.mapping_status!=='resolved_proxy'){host.append(el('p','此字 IPA 映射未解决：'+note.raw_v4c_tokens.join(' ')+'。可确认字的位置及下面整字可听状态，并在页面末尾解释符号；不会自动补IPA。','error'));const controls=el('div');[['full','这个字完整可听'],['partial','只听到这个字一部分'],['uncertain','无法确定']].forEach(([value,text])=>{const b=button(text,()=>{selected.whole_note_audibility=value;changed();for(const v of controls.children)v.className=v===b?'active':'';},selected.whole_note_audibility===value?'active':'');controls.append(b);});host.append(controls);return;}
  host.append(button('本字全部完整可听',()=>{selected.phones=note.phones.map((_,i)=>({phone_index:i,edge_status:'full'}));changed();preserve(()=>renderEdge(side));}));
  note.phones.forEach((phone,index)=>{const row=el('div',undefined,'phone');row.append(el('strong','/'+phone+'/'));const state=selected.phones.find(p=>p.phone_index===index)?.edge_status||'';const controls=el('div');
   [['','未判断'],['full','完整可听'],['partial','部分可听'],['absent','完全听不到'],['uncertain','不确定']].forEach(([value,text])=>{const b=button(text,()=>{selected.phones=selected.phones.filter(p=>p.phone_index!==index);if(value)selected.phones.push({phone_index:index,edge_status:value});selected.phones.sort((a,b)=>a.phone_index-b.phone_index);changed();for(const sibling of controls.children)sibling.className=sibling===b?'active':'';},state===value?'active':'');controls.append(b);});row.append(controls);host.append(row);
  });
 }
 function renderEdges(){renderRange();renderEdge('start');renderEdge('end');}
 $('search').addEventListener('input',()=>preserve(renderCatalog));$('clear-search').addEventListener('click',()=>{$('search').value='';preserve(renderCatalog);});
 $('play-range').addEventListener('click',()=>playNotes(S.selected(packet,draft)));$('stop').addEventListener('click',stop);
 $('reviewer').addEventListener('input',()=>save());$('save').addEventListener('click',()=>save(true));
 $('status').addEventListener('change',()=>{draft.status=$('status').value;changed();});$('comment').addEventListener('input',()=>{draft.comment=$('comment').value;save();});
 $('affirm').addEventListener('change',()=>{draft.sequence_confirmed=$('affirm').checked;save();});
 $('export').addEventListener('click',()=>{try{save();const payload=F.exportAll(batch,all,$('reviewer').value);const blob=new Blob([JSON.stringify(payload,null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='final-test-review-answers-'+batch.packet_sha256.slice(0,12)+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);save();const waiting=batch.items.filter(i=>F.unresolvedNotes(batch,all.answers[i.window_id]).length).map(i=>i.window_id);notice('已合并导出六段记录和五项符号说明。'+(waiting.length?' '+waiting.join('、')+' 的映射仍待解决，已保留全部作答，暂不评分。':'')+' 请把 JSON 发回。');}catch(e){notice(e.message,true);}});
 $('import').addEventListener('change',async event=>{try{const file=event.target.files?.[0];if(!file)return;const payload=JSON.parse(await file.text());all=F.restore(batch,payload);stop();draft=all.answers[current];$('reviewer').value=payload.reviewer;$('status').value=draft.status;$('comment').value=draft.comment;$('affirm').checked=draft.sequence_confirmed;save();preserve(()=>{renderWindows();renderCatalog();renderEdges();renderMapping();});notice('六段答案已导入。');}catch(e){notice('导入失败：'+e.message,true);}finally{event.target.value='';}});
 function renderMapping(){
  const box=$('mapping');box.replaceChildren();
  for(const q of batch.mapping_questions){
   const row=all.mapping_responses.find(m=>m.token===q.token),card=el('section');card.append(el('h3','符号 '+q.token+' · '+q.contexts.length+'处'));
   card.append(el('p',q.contexts.map(c=>c.song_name+'「'+c.lyric+'」 '+c.raw_v4c_tokens.join(' ')).join('；')));
   const examples=el('div');for(const c of q.contexts.slice(0,4)){const song=batch.songs.find(s=>s.song_id===c.song_id);examples.append(button('▶ 听合成参考「'+c.lyric+'」',()=>play(song.rendered_preview,Math.max(0,c.start_sec-.15),Math.min(song.rendered_preview.duration_sec,c.end_sec+.15)-Math.max(0,c.start_sec-.15),'合成符号示例，仅作说明辅助')));}
   const select=el('select');for(const [v,t] of [['pending','暂不确定'],['described','已说明其含义']]){const o=el('option',t);o.value=v;select.append(o);}select.value=row.status;
   const text=el('textarea');text.rows=3;text.placeholder='你在校准时用这个符号表示什么发音？可写IPA、自己的符号约定、相同读音的字或详细说明；不必猜。';text.value=row.description;
   select.addEventListener('change',()=>{row.status=select.value;save();});text.addEventListener('input',()=>{row.description=text.value;save();});card.append(examples,select,text);box.append(card);
  }
 }
 try{const saved=JSON.parse(localStorage.getItem(key)||'null');if(saved&&saved.packet_sha256===batch.packet_sha256&&saved.all&&batch.items.every(i=>saved.all.answers?.[i.window_id]?.edges)&&batch.items.some(i=>i.window_id===saved.current)){all=saved.all;current=saved.current;packet=F.packetFor(batch,current);draft=all.answers[current];$('reviewer').value=saved.reviewer||'';$('status').value=draft.status;$('comment').value=draft.comment;$('affirm').checked=draft.sequence_confirmed;}}catch(_){notice('草稿恢复失败，可导入同页答案。',true);}
 renderWindows();renderCatalog();renderEdges();renderMapping();
})();
