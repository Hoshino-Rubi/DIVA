using Any2Xsampa_Desktop.Models;
using Grpc.Net.Client;

namespace Any2Xsampa_Desktop.Services;

public sealed class GrpcAudioInferenceService : IAudioInferenceService
{
    private readonly Grpc.MikuV4CProductDraftService.MikuV4CProductDraftServiceClient _client;

    public GrpcAudioInferenceService(GrpcChannel channel)
    {
        _client = new Grpc.MikuV4CProductDraftService.MikuV4CProductDraftServiceClient(channel);
    }

    public async Task<DraftVsqxResult> InferDraftVsqxAsync(
        DraftVsqxRequest request,
        CancellationToken cancellationToken = default)
    {
        var response = await _client.InferDraftVsqxAsync(
            new Grpc.InferDraftVsqxRequest
            {
                AudioPath = request.AudioPath,
                TemplatePath = request.TemplatePath,
                OutputRoot = request.OutputRoot,
                Lyrics = request.Lyrics,
                JyutpingOverride = request.JyutpingOverride,
                AsrTranscriptPath = request.AsrTranscriptPath,
                PitchAnalysisPath = request.PitchAnalysisPath,
                Qwen3AsrModel = request.Qwen3AsrModel,
                PitchesCsv = request.PitchesCsv,
                Tempo = request.Tempo,
                UseFakeAsr = request.UseFakeAsr,
                Timestamps = request.Timestamps,
                Language = request.Language,
                Context = request.Context,
                Prefix = request.Prefix,
            },
            cancellationToken: cancellationToken);

        return new DraftVsqxResult(
            response.Ok,
            response.SchemaVersion,
            response.ErrorKind,
            response.Errors.ToArray(),
            response.VsqxPath,
            response.DraftPath,
            response.PhonemeLatticePath,
            response.ExportSummaryPath,
            response.AudioProductSummaryPath,
            response.LyricSource,
            response.ManualVocaloidImportRequired,
            response.UserJyutpingRequired,
            response.ProductSummaryJson);
    }
}
