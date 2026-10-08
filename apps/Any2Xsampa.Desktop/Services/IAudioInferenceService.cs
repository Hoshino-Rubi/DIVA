using Any2Xsampa_Desktop.Models;

namespace Any2Xsampa_Desktop.Services;

public interface IAudioInferenceService
{
    Task<DraftVsqxResult> InferDraftVsqxAsync(DraftVsqxRequest request, CancellationToken cancellationToken = default);
}
