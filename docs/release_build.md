# Building pre-1.0.1a

This reliability release uses the existing weights. Engineering acceptance runs no model forwards or training updates.

## Checks and explicit inputs

~~~text
python -m pip install -r tools/requirements/release-ci.txt
python tools/check_windows_release.py --output .state/checks-<unique-id>
~~~

Publish the desktop for x64, self-contained and unpackaged, with trimming and ReadyToRun disabled. Bind every published file to the code review.

The portable builder requires --desktop, --review and --inputs in addition to new output directories. Input roots are resolved relative to the private manifest's artifact_root. No historical paths are built into the tool. See docs/distribution/build-inputs.example.json.

The private input manifest binds model snapshot locks, the existing candidate freeze, and complete SHA256 inventories of both audited runtime directories. The review binds current product source, the input manifest, published desktop files, and passing verification artifacts. Changed inputs invalidate it. The builder reads existing files; it never downloads models, installs dependencies or chooses a checkpoint. Do not commit private artifact locations or local test output.

## Publication

Create one versioned application root in the ZIP. Split with tools/split_portable_release.py and embed its manifest in Setup with -p:ReleaseManifestPath=<absolute-path>. Install and verify in a new directory.

Finalize attachments with tools/finalize_release_assets.py and an explicit JSON array of file names. Missing or extra files cause failure. SHA256SUMS covers only actual attachments and excludes itself. Keep maintainer notes outside the asset directory.

Release body links use versioned source URLs. README.md stays English with a relative link to README_CN.md. Merge this branch into main through a reviewed PR to update the default homepage. Do not rewrite pre-1.0.0, change visibility or select a license implicitly.

## Signing

This build is unsigned; no trusted signing identity is configured. For a future signed release, sign and timestamp desktop executables before hashing and archiving. Sign Setup after embedding its manifest, verify it, then regenerate checksums. The finalizer supports --require-installer-signature --signtool <path>; this checks the installer only. Never commit a private key or pass a PFX password on the command line. Self-signing is not a substitute for trust.

## Limits

Local checks use generated PCM and fixture evidence in a relocated directory. They validate failure handling, transport and VSQX serialization. They do not certify recognition accuracy, another machine's GUI or a GitHub-hosted workflow run. The full historical research suite additionally needs unpublished local artifacts.
