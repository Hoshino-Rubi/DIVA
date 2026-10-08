using Any2Xsampa_Desktop.Services;
using System.Text.Json;
using System.Security.Cryptography;

if(args.Length==4 && args[0]=="--portable-setup")
{
    using var portable=new LocalTaskService(args[1]);
    var before=await portable.CallAsync("model_setup_status",new{});
    if(!before.GetProperty("enabled").GetBoolean())throw new Exception("Installer unavailable");
    int progressCount=0;
    portable.Progress+=_=>progressCount++;
    var done=await portable.CallAsync("install_models",new{offline_dir=args[2]});
    if(!done.GetProperty("ready").GetBoolean())throw new Exception("Offline installation incomplete");
    var models=await portable.CallAsync("models",new{});
    if(!models.GetProperty("models").EnumerateArray().All(x=>x.GetProperty("available").GetBoolean()))
        throw new Exception("Portable runtime verification failed");
    var note=await portable.CallAsync("note_runtime",new{});
    if(!note.GetProperty("available").GetBoolean())throw new Exception("GAME not ready");
    var output=JsonSerializer.Serialize(new{status="passed",transport="CSharp_stdio",
        installation_ready=true,model_forwards=0,progressCount,models,note});
    await File.WriteAllTextAsync(args[3],output);
    Console.WriteLine(output);
    return;
}

if(args.Length>=3 && args[0]=="--validate-vsqx")
{
    var settings=new System.Xml.XmlReaderSettings{ValidationType=System.Xml.ValidationType.Schema};
    settings.Schemas.Add(null,args[1]);
    var errors=new List<string>();
    settings.ValidationEventHandler+=(_,e)=>{if(e.Severity==System.Xml.Schema.XmlSeverityType.Error)errors.Add(e.Message);};
    foreach(string vsqxFile in args.Skip(2))
    {
        errors.Clear();
        using(var reader=System.Xml.XmlReader.Create(vsqxFile,settings)){while(reader.Read()){};}
        if(errors.Count>0)throw new Exception(vsqxFile+": "+string.Join(" | ",errors));
        Console.WriteLine(JsonSerializer.Serialize(new{file=vsqxFile,status="xsd_valid",errors=errors.Count}));
    }
    return;
}
if(args.Length==5 && args[0]=="--game-vsqx")
{
    using var client=new LocalTaskService(args[1]);
    byte[] before=await File.ReadAllBytesAsync(args[2]);
    var response=await client.CallAsync("build_score",new{
        result_path=args[2],backend="game",notes_path=args[3],output_dir=args[4],tempo=120});
    if(response.GetProperty("model_forwards").GetInt32()!=0)throw new Exception("Saved notes triggered inference");
    string score=response.GetProperty("score_path").GetString()!;
    string generatedVsqx=response.GetProperty("vsqx_path").GetString()!;
    string copy=Path.Combine(args[4],"exported.vsqx");
    await client.CallAsync("export_score",new{
        score_path=score,score_sha256=response.GetProperty("score_sha256").GetString(),output=copy});
    if(!File.ReadAllBytes(copy).SequenceEqual(File.ReadAllBytes(generatedVsqx)))throw new Exception("Export changed VSQX");
    if(!before.SequenceEqual(await File.ReadAllBytesAsync(args[2])))throw new Exception("Phone evidence changed");
    var xml=System.Xml.Linq.XDocument.Load(copy);
    var ns=xml.Root!.Name.Namespace;
    int count=xml.Descendants(ns+"note").Count();
    if(count!=response.GetProperty("note_count").GetInt32())throw new Exception("Note count differs");
    Console.WriteLine(JsonSerializer.Serialize(new{status="passed",transport="CSharp_stdio",
        note_count=count,model_forwards=0,vsqx=copy,source_unchanged=true}));
    return;
}

if(args.Length!=4)throw new ArgumentException("config fixture audio output");
using var service=new LocalTaskService(args[0]);
service.Log+=line=>Console.Error.WriteLine(line);
var caps=await service.CallAsync("capabilities",new{});
if(caps.GetProperty("transport").GetString()!="stdio")throw new Exception("Not stdio");
var r=await service.CallAsync("replay",new{fixture=args[1],audio=args[2],duration=2,output_dir=args[3]});
if(r.GetProperty("mode").GetString()!="replay")throw new Exception("Fixture disguised as real");
string file=r.GetProperty("result_path").GetString()!;
byte[] original=await File.ReadAllBytesAsync(file);
using var result=JsonDocument.Parse(original);
var preview=await service.CallAsync("preview_xsampa",new{result_path=file,result_sha256=Convert.ToHexString(SHA256.HashData(original)).ToLowerInvariant()});
if(preview.GetProperty("xsampa").GetArrayLength()!=result.RootElement.GetProperty("segments").GetArrayLength()
   || preview.GetProperty("model_forwards").GetInt32()!=0
   || preview.GetProperty("source_modified").GetBoolean())throw new Exception("Invalid read-only export preview");
var edits=result.RootElement.GetProperty("segments").EnumerateArray().Select((s,i)=>new{index=i,ipa=s.GetProperty("normalized_ipa").GetString()}).ToArray();
await service.CallAsync("draft",new{result_path=file,result_sha256=Convert.ToHexString(SHA256.HashData(original)).ToLowerInvariant(),edits});
await service.CallAsync("export",new{result_path=file,format="ipa",output=Path.Combine(args[3],"manual-ipa.txt"),use_draft=true});
if(!original.SequenceEqual(await File.ReadAllBytesAsync(file)))throw new Exception("Machine output changed");
try
{
 await service.CallAsync("recognize",new{audio=args[2],model="not-configured",duration=1,output_dir=Path.Combine(args[3],"missing-model")});
 throw new Exception("Missing model unexpectedly succeeded");
}
catch(InvalidOperationException){ }
try
{
 await service.CallAsync("recognize-whole",new{audio=args[2],model="not-configured",device="cuda"});
 throw new Exception("Missing whole-file model unexpectedly succeeded");
}
catch(InvalidOperationException){ }
using(var c=new CancellationTokenSource())
{
 c.Cancel();
 try{await service.CallAsync("doctor",new{},c.Token);throw new Exception("Cancellation ignored");}
 catch(OperationCanceledException){}
}
await service.CallAsync("unload",new{});
Console.WriteLine(JsonSerializer.Serialize(new{status="passed",transport="stdio",model_forwards=0,checks=new[]{"handshake","UTF8_paths","replay","xsampa_preview","draft","export","missing_model_fail_closed","missing_whole_model_fail_closed","cancellation","unload"}}));
