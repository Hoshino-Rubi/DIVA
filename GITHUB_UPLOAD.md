# Publishing Any2Xsampa to GitHub

This directory is an independently prepared source copy, not the original research workspace.
It excludes model weights, virtual environments, caches, personal audio and manual projects.
Do not run `git add .` in the original research directory.

## Source repository

The existing local copy has a `main` branch, committed history and an `origin` remote. This task staged the new changes but did not create a new commit, authenticate or push. For that checkout, use its existing history/remote; do not recreate the repository.
If using the source ZIP, extract it and initialize Git:

~~~bat
git init -b main
git config user.name "YOUR_GITHUB_NAME"
git config user.email "YOUR_COMMIT_EMAIL"
git add .
git commit -m "Initial source and desktop preview"
gh auth login --hostname github.com --git-protocol https --web
gh repo create any2xsampa --private --source=. --remote=origin --push
~~~

Replace the identity and repository name. Use `--public` instead of `--private` only when you intend to publish publicly.
For an existing empty repository, add its URL as `origin` and push `main` instead of running `gh repo create`.
Do not force-push over an existing repository's history. Never paste credentials into code or chat.

If your local network needs a proxy, use its actual address:

~~~bat
set HTTPS_PROXY=http://127.0.0.1:18080
set HTTP_PROXY=http://127.0.0.1:18080
~~~

These commands are instructions only; no remote publication has been performed.

## Desktop Release

The separately prepared `Any2Xsampa-Release-assets-20261008` directory is ready for attachment upload after its verification completes.
Create a preview Release and upload every file from that directory. Do not recompress the directory or upload the development workspace.
The installer and its three ZIP volumes must all be provided. Optional offline model volumes remain Release attachments, **not Git files**.

Use `RELEASE_README.md` for the English Release description. It links to `RELEASE_README_CN.md`.
The application downloads pinned weights directly from their upstream sources; uploading offline model files is not a prerequisite for automatic online setup.
See `UPLOAD_TO_GITHUB.md` in the attachment directory for the asset layout and `SHA256SUMS.txt` for hashes.

Unlike the earlier workspace-bound build, this preview includes isolated Python runtimes and self-contained .NET/Windows App SDK files.
Local installation, relocation and saved-result export have been verified; signing and another-machine GUI verification remain incomplete.
No formal joint transcription accuracy is claimed.

## Licenses and development

The owner has not selected a license for the application's own source yet.
Third-party notices do not license your code on your behalf. GAME weights use CC BY-NC-SA 4.0, not the MIT license of GAME code.
See [third-party notices](docs/distribution/THIRD_PARTY_NOTICES.md).

The historical `requirements.lock.txt` records the full development environment, not minimal end-user requirements.
Research tools can require caches omitted from this repository. Normal desktop and CLI operation use the shared local stdio engine.

Internal audit records contain local paths and are intentionally not included in the source archive.
