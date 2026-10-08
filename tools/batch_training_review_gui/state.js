(function(root){
  'use strict';
  const schema='batch_training_review_answers_v1';
  const requireValue=(test,message)=>{if(!test)throw new Error(message);};
  function empty(item,restorePrevious=true){
    const draft={choice:item.correspondence_preconfirmed?'A':'',status:'',comment:'',sequence_confirmed:false,edges:{start:{note_id:null,phones:[]},end:{note_id:null,phones:[]}}};
    const prior=restorePrevious&&item.previous_answer;
    if(prior&&prior.status==='confirmed'){
      draft.choice=prior.choice;draft.comment=prior.comment;
      for(const side of ['start','end'])draft.edges[side]={note_id:prior.edges[side].note_id,phones:structuredClone(prior.edges[side].phones)};
    }
    return draft;
  }
  function setChoice(item,draft,value){return {...empty(item,false),choice:value,comment:draft.comment};}
  function validate(packet,payload){
    requireValue(payload&&payload.schema_version===schema&&payload.packet_sha256===packet.packet_sha256,'答案与当前批次不匹配');
    requireValue(typeof payload.reviewer==='string'&&payload.reviewer.trim()&&payload.reviewer.length<=120,'请填写审核人');
    requireValue(Array.isArray(payload.answers)&&payload.training_started===false,'答案格式无效');
    const seen=new Set();
    for(const answer of payload.answers){
      const item=packet.items.find(i=>i.review_id===answer.review_id);
      requireValue(item&&!seen.has(answer.review_id),'未知或重复项目');seen.add(answer.review_id);
      requireValue(['confirmed','uncertain','rejected'].includes(answer.status),'请选择结论');
      requireValue(typeof answer.comment==='string'&&answer.comment.length<=4000,'补充说明过长');
      if(answer.status!=='confirmed'){
        requireValue(answer.choice===null&&answer.sequence_confirmed===false&&answer.edges&&Object.keys(answer.edges).length===0,'待定或否决不能携带音素确认');continue;
      }
      const choice=item.alternatives.find(a=>a.choice_id===answer.choice);
      requireValue(choice&&answer.sequence_confirmed===true,'请明确选择对应并勾选连续范围确认');
      requireValue(!item.correspondence_preconfirmed||answer.choice==='A','已有对应不能静默替换');
      requireValue(answer.edges&&Object.keys(answer.edges).sort().join(',')==='end,start','需要首尾两端记录');
      for(const side of ['start','end']){
        const edge=answer.edges[side],note=choice.catalog.find(n=>n.note_id===edge.note_id);
        requireValue(edge.review_id===item.review_id+(side==='start'?'-S':'-E')&&edge.status==='confirmed'&&edge.observation_schema==='phone_audibility_v2','边缘记录无效');
        requireValue(note&&note.mapping_status==='resolved_proxy'&&note.phones.length,'请选择映射已解决的边缘字；无法确定时标为待定');
        requireValue(edge.audible_identity_confirmed===true&&typeof edge.statement==='string'&&edge.statement.trim()&&edge.statement.length<=4000,'缺少边缘确认');
        requireValue(Array.isArray(edge.phones)&&edge.phones.length===note.phones.length,'每个音素都需记录，包括不确定');
        edge.phones.forEach((p,i)=>requireValue(p.phone_index===i&&['full','partial','absent','uncertain'].includes(p.edge_status),'音素状态或索引无效'));
      }
      const ids=choice.catalog.map(n=>n.note_id);
      requireValue(ids.indexOf(answer.edges.start.note_id)<=ids.indexOf(answer.edges.end.note_id),'首字不能位于末字之后');
    }
    return structuredClone(payload);
  }
  function commit(packet,item,draft,reviewer){
    let answer={review_id:item.review_id,status:draft.status,choice:null,comment:draft.comment,sequence_confirmed:false,edges:{}};
    if(draft.status==='confirmed'){
      answer.choice=draft.choice;answer.sequence_confirmed=draft.sequence_confirmed;
      for(const side of ['start','end'])answer.edges[side]={review_id:item.review_id+(side==='start'?'-S':'-E'),status:'confirmed',
        observation_schema:'phone_audibility_v2',note_id:draft.edges[side].note_id,phones:structuredClone(draft.edges[side].phones),
        audible_identity_confirmed:draft.sequence_confirmed,
        statement:'我确认所选字为原唱正式片段内的'+(side==='start'?'首字':'末字')+'，逐音素记录实际可听状态；内部为连续参考，缺失原因不推断。'};
    }
    return validate(packet,{schema_version:schema,packet_sha256:packet.packet_sha256,reviewer,answers:[answer],training_started:false}).answers[0];
  }
  function draftFromAnswer(item,answer){
    const draft=empty(item);draft.status=answer.status;draft.comment=answer.comment;
    if(answer.status==='confirmed'){
      draft.choice=answer.choice;draft.sequence_confirmed=true;
      for(const side of ['start','end'])draft.edges[side]={note_id:answer.edges[side].note_id,phones:structuredClone(answer.edges[side].phones)};
    }
    return draft;
  }
  function exportAnswers(packet,answers,reviewer){return validate(packet,{schema_version:schema,packet_sha256:packet.packet_sha256,reviewer,
    exported_at:new Date().toISOString(),answers,training_started:false});}
  const api={empty,setChoice,validate,commit,draftFromAnswer,exportAnswers};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.BatchReviewState=api;
})(typeof globalThis!=='undefined'?globalThis:this);
