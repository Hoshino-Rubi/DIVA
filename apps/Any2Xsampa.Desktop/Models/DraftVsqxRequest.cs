namespace Any2Xsampa_Desktop.Models;

public sealed record DraftVsqxRequest(
    string AudioPath,
    string TemplatePath,
    string OutputRoot,
    string Lyrics,
    string JyutpingOverride,
    string AsrTranscriptPath,
    string PitchAnalysisPath,
    string Qwen3AsrModel,
    string PitchesCsv,
    double Tempo,
    bool UseFakeAsr,
    bool Timestamps,
    string Language,
    string Context,
    string Prefix);
