# Any2Xsampa pre-1.0.1a

[简体中文](https://github.com/Hoshino-Rubi/any2xsampa/blob/pre-1.0.1a/docs/releases/pre-1.0.1a_CN.md)

Reliability update for the Windows x64 research preview.

- Added local diagnostic ZIP export, including startup failures, with allowlisted error codes, expected model hashes, file sizes and resource counters.
- Diagnostics exclude audio, predictions, raw logs, usernames, full paths and proxy credentials. Nothing uploads automatically.
- Fixed persistent worker log ownership; tasks now link to their worker session.
- Added Windows product regression, relocated stdio/VSQX fixture checks and SHA-bound explicit build inputs.
- Public checksums cover the actual attachment list.

Download **Any2Xsampa-Setup.exe** and the three **Any2Xsampa-Windows-x64-pre-1.0.1a.zip.001 / .002 / .003** files into one folder. Run Setup and choose a new writable location. Python and .NET are bundled.

First launch downloads the same pinned PhoneticXeus/GAME weights as pre-1.0.0. The optional four offline model assets are unchanged and can be reused.

[Installation and CLI](https://github.com/Hoshino-Rubi/any2xsampa/blob/pre-1.0.1a/README.md) · [Chinese guide](https://github.com/Hoshino-Rubi/any2xsampa/blob/pre-1.0.1a/README_CN.md)

Unsigned preview. No new training or accuracy improvement is claimed. GAME weights retain CC BY-NC-SA 4.0 restrictions. Formal joint pitch/onset/offset accuracy and another computer's GUI launch remain unverified.
