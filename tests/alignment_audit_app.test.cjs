/* Offline DOM fixture for the actual app script; no browser or desktop control. */
"use strict";
const {test}=require("node:test"),assert=require("node:assert/strict");
const fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const assets=path.join(__dirname,"../tools/alignment_audit_gui");
const decode=s=>s.replace(/&quot;/g,'"').replace(/&#39;/g,"'").replace(/&lt;/g,"<").replace(/&gt;/g,">").replace(/&amp;/g,"&");
const voids=new Set(["meta","input","img","br","hr","source","link"]);
function match(e,selector){
  if(selector.startsWith("."))return (e.attrs.class||"").split(" ").includes(selector.slice(1));
  const m=selector.match(/^\[([^=\]]+)(?:="([^"]*)")?\]$/);
  return m ? Object.hasOwn(e.attrs,m[1])&&(m[2]===undefined||e.attrs[m[1]]===m[2]) : e.tagName===selector.toUpperCase();
}
class Element{
  constructor(tag,attrs={}){this.tagName=tag.toUpperCase();this.attrs=attrs;this.children=[];this._text="";this.style={};this.events={};this.paused=true;this.duration=10;this.currentTime=0;this.files=[];
    this.dataset=Object.fromEntries(Object.entries(attrs).filter(([k])=>k.startsWith("data-")).map(([k,v])=>[k.slice(5).replace(/-([a-z])/g,(_,x)=>x.toUpperCase()),v]));
    this.classList={toggle:(name,on)=>{const set=new Set((this.attrs.class||"").split(" ").filter(Boolean));on?set.add(name):set.delete(name);this.attrs.class=[...set].join(" ");}};
  }
  get id(){return this.attrs.id||"";}
  setAttribute(k,v){this.attrs[k]=v;}
  get value(){if(this._value!==undefined)return this._value;if(this.tagName==="SELECT"){const opts=this.children.filter(x=>x.tagName==="OPTION");return(opts.find(x=>Object.hasOwn(x.attrs,"selected"))||opts[0])?.value||"";}return this.attrs.value??this._text;}
  set value(v){this._value=String(v);}
  get textContent(){return this._text+this.children.map(x=>x.textContent).join("");}
  set textContent(v){this.children=[];this._text=String(v);}
  get innerHTML(){const serialize=e=>'<'+e.tagName.toLowerCase()+Object.entries(e.attrs).map(([k,v])=>' '+k+'="'+v+'"').join('')+'>'+e._text+e.children.map(serialize).join('')+'</'+e.tagName.toLowerCase()+'>';return this._html??(this._text+this.children.map(serialize).join(''));}
  set innerHTML(html){this._html=html;this.children=[];this._text="";parseInto(this,html);}
  querySelectorAll(s){const out=[];for(const c of this.children){if(match(c,s))out.push(c);out.push(...c.querySelectorAll(s));}return out;}
  querySelector(s){return this.querySelectorAll(s)[0]||null;}
  closest(s){let e=this;while(e){if(match(e,s))return e;e=e.parent;}return null;}
  addEventListener(name,fn){(this.events[name]??=[]).push(fn);}
  dispatch(name,extra={}){const event={target:this,preventDefault(){},...extra};for(const fn of this.events[name]||[])fn(event);return this["on"+name]?.(event);}
  click(){return this.dispatch("click");}
  pause(){this.paused=true;}
  play(){this.paused=false;this.onplay?.();return Promise.resolve();}
  getBoundingClientRect(){return{left:0,width:1200};}
}
function parseInto(parent,html){
  const stack=[parent];const tokens=html.match(/<!--[^]*?-->|<![^>]*>|<\/?[a-zA-Z][^>]*>|[^<]+/g)||[];
  for(const token of tokens){
    if(token.startsWith("<!"))continue;
    if(token.startsWith("</")){const name=token.slice(2,-1).trim().toUpperCase();while(stack.length>1){if(stack.pop().tagName===name)break;}continue;}
    if(token.startsWith("<")){
      const m=token.match(/^<([\w-]+)([^]*?)>$/);if(!m)continue;
      const attrs={};for(const a of m[2].matchAll(/([\w:-]+)(?:="([^"]*)"|='([^']*)')?/g))attrs[a[1]]=decode(a[2]??a[3]??"");
      const e=new Element(m[1],attrs),p=stack.at(-1);e.parent=p;p.children.push(e);if(!voids.has(m[1].toLowerCase()))stack.push(e);
    }else stack.at(-1)._text+=decode(token);
  }
}
function fixture(){
  const row=(id,rejected,kind)=>({review_id:id,kind,song_id:kind==="boundary"?"shalong":"like-you",song_name:kind==="boundary"?"沙龙":"喜欢你",
    packet_sha256:"b".repeat(64),source_example_id:"original01",source_audio_sha256:"c".repeat(64),source_vsqx_sha256:"d".repeat(64),start_sec:1,end_sec:11,
    original_cut_sec:2,cut_in_preview_sec:1.25,side:"start",human:{status:rejected?"rejected":"human_confirmed",reviewer:"Admin",review_statement:rejected?"错位":"全部通过",phones:[]},
    machine:{status:rejected?"machine_aligned":"needs_review",confidence:{score:.7}},state:{human_rejected:rejected,human_machine_conflict:rejected,machine_flagged:!rejected},
    warnings:[],notes:[{note_id:"n1",lyric:"笑",phones:["s","a"],engine_tokens:["s","a"],mapping_status:"resolved_proxy"}],nearby_note_ids:[],candidate_references:[],mapping_resolved_notes:1,
    media:["original","rendered"].map(role=>({role,duration_sec:kind==="boundary"?2.5:10,audio_file:role+".wav",waveform_file:role+".png"}))});
  return{bundle_id:"a".repeat(64),built_at:"2026-09-15T00:00:00Z",sources:{reference:{sha256:"f".repeat(64)}},
    summary:{items:3,human_rejected:1,human_machine_conflicts:1,training_text_approvals:1},rows:[row("C01",true,"training"),row("C03",false,"training"),row("B01",false,"boundary")]};
}
function setup(){
  const data=fixture(),doc=new Element("document");parseInto(doc,fs.readFileSync(path.join(assets,"index.html"),"utf8"));
  doc.getElementById=id=>{const search=e=>e.id===id?e:e.children.map(search).find(Boolean);return search(doc)||null;};
  doc.createElement=tag=>new Element(tag);doc.getElementById("audit-data").textContent=JSON.stringify(data);
  const storage=new Map(),capture={};
  const context=vm.createContext({document:doc,window:{confirm:()=>true,addEventListener(){}},localStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v)},
    console,Blob,URL:{createObjectURL:b=>{capture.blob=b;return"blob:fixture";},revokeObjectURL(){}},setTimeout:()=>1,clearTimeout(){}});
  vm.runInContext(fs.readFileSync(path.join(assets,"audit_core.js"),"utf8"),context);
  vm.runInContext(fs.readFileSync(path.join(assets,"app.js"),"utf8"),context);
  return{doc,data,storage,capture,$:id=>doc.getElementById(id)};
}
test("actual GUI script boots, defaults to non-rejected row and renders audio",()=>{
  const {$}=setup();assert.match($("detail").textContent,/C03/);assert.equal($("item-list").querySelectorAll("[data-select]").length,2);
  assert.equal($("detail").querySelectorAll("audio").length,2);
});
test("actual form saves Unicode, exports and restores through import handler",async()=>{
  const {$,storage,capture}=setup();$("detail").querySelector('[data-decision="boundary_issue"]').click();
  $("comment").value="笑字被截断 ŋ";$("comment").dispatch("input");$("save-item").click();
  assert.equal(storage.size,1);$("export-button").click();
  const payload=JSON.parse(await capture.blob.text());assert.equal(payload.decisions[0].comment,"笑字被截断 ŋ");assert.equal(payload.creates_phone_reference,false);
  const target=setup();target.$("import-file").files=[{size:100,text:async()=>JSON.stringify(payload)}];
  await target.$("import-file").dispatch("change");assert.equal(target.$("comment").value,"笑字被截断 ŋ");
});
test("actual filters show conflict and preserve rejected history",()=>{
  const {$}=setup();$("filter").value="conflict";$("filter").dispatch("change");
  const buttons=$("item-list").querySelectorAll("[data-select]");assert.equal(buttons.length,1);buttons[0].click();
  assert.match($("detail").textContent,/人工已判错位/);assert.ok($("revise"));
});
test("switching row pauses existing player and boundary audition does not change source",async()=>{
  const {$,data}=setup();const before=JSON.stringify(data);await $("play").click();
  const oldAudio=$("detail").querySelector('[data-audio="original"]');assert.equal(oldAudio.paused,false);
  $("item-list").querySelector('[data-select="B01"]').click();assert.equal(oldAudio.paused,true);
  $("cut-after").click();assert.equal($("loop-start").value,"1.25");assert.equal(JSON.stringify(data),before);
});
