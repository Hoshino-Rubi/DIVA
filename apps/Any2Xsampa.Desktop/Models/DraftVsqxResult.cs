namespace Any2Xsampa_Desktop.Models;

public sealed record DraftVsqxResult(
    bool Ok,
    string SchemaVersion,
    string ErrorKind,
    IReadOnlyList<string> Errors,
    string VsqxPath,
    string DraftPath,
    string PhonemeLatticePath,
    string ExportSummaryPath,
    string AudioProductSummaryPath,
    string LyricSource,
    bool ManualVocaloidImportRequired,
    bool UserJyutpingRequired,
    string ProductSummaryJson);
