# DIVA branding migration / 品牌迁移

## Scope / 迁移范围

Product name: **DIVA (Dialect Intonation & Vocal Automation)**, formerly Any2Xsampa.

The first stage changed display branding in source. The second stage migrates the repository to `Hoshino-Rubi/DIVA` and prepares DIVA v1.0.2. Display changes cover the README, desktop and legacy GUI titles, installer captions, Windows display metadata, CLI help and default names of newly exported VSQX/diagnostic files. No additional dependency, model training or inference is required. Existing user files are not renamed.

第二阶段已获授权进行仓库改名和新版发布：仓库改为 `Hoshino-Rubi/DIVA`，版本为 v1.0.2。历史 v1.0.1 继续使用 Any2Xsampa 名称，不覆盖旧安装包。公开可下载状态以 GitHub Release 为准；不能把构建成功视为已发布。

## Compatibility boundary / 保留的兼容标识

| Surface | Preserved identifier |
| --- | --- |
| GitHub repository and new release URLs | `Hoshino-Rubi/DIVA`; preserve legacy `Hoshino-Rubi/any2xsampa` redirects and historical pinned URLs |
| Upstream model URLs | Existing revision-pinned PhoneticXeus and GAME URLs |
| Python distribution, imports and CLI | `any2xsampa`, `any2xsampa-api`, `any2xsampa-gui`, `any2xsampa-grpc`, `any2xsampa.cmd` |
| Desktop/installer executables and project paths | `Any2Xsampa.Desktop.exe`, `Any2Xsampa-Setup`, `apps/Any2Xsampa.*` |
| Windows package/signing identity | Existing identity GUID and `CN=any2xsampa` |
| Settings, logs, results and model caches | `%LOCALAPPDATA%\Any2Xsampa`, legacy local state and GUI history paths |
| Environment variables | Existing `ANY2XSAMPA_*` names, including `ANY2XSAMPA_DATA_HOME` |
| Local/gRPC protocols, JSON and install receipts | `any2xsampa.local.v1`, `any2xsampa.v1`, existing schemas and `PhoneSegment` contract |
| Published artifacts and evidence | v1.0.1 and older tags, filenames, manifests, hashes and release notes |

Do not globally replace `any2xsampa` with `diva`. Display branding must not change imports, storage discovery, protocol names, trusted download origins or historical evidence. Existing model and acoustic evaluations do not become more accurate when the product name changes.

## Product description / 对外介绍边界

Use the introduction in the bilingual README. Cantonese/Hokkien coverage is a product goal, not a new quality certification. Do not claim perfect alignment, universal dialect coverage, end-to-end synthesized vocal audio rendering, or an exclusively custom-trained model based on tailored VSQX datasets without evidence for the shipped pipeline. IPA / phones remain the intermediate representation; X-SAMPA remains an export layer.

## Later migration / 后续迁移

Repository renaming, remote URL edits and v1.0.2 release preparation are part of the second stage. Local workspace moves and new CLI aliases are deferred. Installer URL validation accepts only the exact DIVA or legacy repository URL plus the bound stable tag; new manifests and update checks use DIVA. Upstream model catalogs remain unchanged. Preserve old release IDs/assets and explicitly verify old download URLs after renaming; do not rely on an assumed redirect. Do not recreate a repository at the old `any2xsampa` name while old installers depend on its redirect.

Code changes invalidate source-bound launch/build reviews for the modified code. v1.0.2 needs new source/input hashes, review and relevant zero-model installation acceptance evidence; the v1.0.1 publication review applies only to the original artifacts. No previous model experiment or acoustic acceptance group is restarted for this branding release.

## Verification / 验证

Run `python -m pytest tests/test_product_branding.py tests/test_local_desktop_layout.py tests/test_local_storage.py tests/test_release_version.py -q -p no:cacheprovider` for the focused gate. The branding checks are also part of `tools/check_windows_release.py`. Build desktop and Setup to validate their edited XAML/C# and metadata; do not publish or run real models as part of this branding change.
