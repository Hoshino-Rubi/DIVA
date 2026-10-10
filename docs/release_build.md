# Building v1.0.1

release-version.json is the only edited release identity. Run python tools/sync_release_version.py to regenerate mirrors; --check fails on drift. Build and diagnostic version values use the generated assembly identity.

## Verification and inputs

Run python tools/check_windows_release.py --output <new-directory>. It performs product tests, installer download/resume/cancel/model-reuse/uninstall checks, local transport and VSQX replay without loading weights. The full historical suite additionally needs private artifacts.

The portable builder requires --desktop, --review, --inputs, --output and --assets. It verifies model locks, existing checkpoints and complete runtime inventories, current source and published desktop binaries before copying. Changed source invalidates the review. Keep private build-input paths out of Git.

Publish the desktop self-contained for win-x64 with trimming and ReadyToRun disabled. Freeze its hashes and passing tests. Keep one current staging directory. Run `python tools/stream_portable_release.py <staging> <assets> --release-tag v1.0.1` to compress directly into 1 GiB volumes without a duplicate full ZIP. This caps archive output at 6 GiB, reserves 30 GiB of free disk space, and verifies each volume SHA256 and the complete ZIP CRC before writing the manifest. Existing volumes cause a stop, not a retry. The v2 manifest embeds the exact release tag, fixed GitHub base URL, sizes and SHA256. Setup never downloads a mutable manifest. Use one temporary installation for acceptance; preserve its evidence and remove generated copies after verification.

Embed the resulting manifest through -p:ReleaseManifestPath=<absolute-path>. Rename the published Setup executable to Any2Xsampa-Setup-v1.0.1.exe. Install and verify in a new path. Installed files receive a hashed receipt; uninstall removes only unchanged listed program files and retains models and user data.

## Product acceptance

No new training or model downloads are part of this release. A separate hash-bound acceptance plan defines one 8-second smoke and one complete training song, bounded by the frozen audio window counts. It uses the installed runtimes, no lyrics/reference inputs, CPU four threads, GPU serial, 7 GiB per-process reserved / 14 GiB whole-card limits and no automatic retries. Count actual encoder and ONNX calls. Zero-inference replay then checks VSQX output against the saved predicted notes; this does not certify the predictions against original-singing ground truth.

## Publication

The automatic installer needs an anonymous public GitHub Release. Private repositories or draft-only assets cannot satisfy that gate; never embed a token. Upload final assets to a new v1.0.1 draft, verify the set and hashes, then publish only after release authorization. An anonymous download probe after publication is required before claiming the hosted online-install path works. Do not change visibility or old releases implicitly.

README.md remains English with README_CN.md linked. Formal software version stability does not establish acoustic accuracy. The application source license remains an owner choice; no overall license is inferred from third-party models. GAME weights retain CC BY-NC-SA 4.0.

## Signing

The current candidate is unsigned; it must be labelled unsigned. If a trusted certificate is provided, sign/timestamp desktop executables before hashing/archiving; sign Setup after embedding its manifest. Regenerate asset checksums after signing. Self-signing is not a trust substitute. The finalizer supports an explicit --installer-name and --require-installer-signature; do not put keys or passwords in the repo.
