# Publishing DIVA v1.0.2

Use the existing source checkout on release-v1.0.2 with remote `https://github.com/Hoshino-Rubi/DIVA.git`. Do not git init again, force-push main, rewrite old tags or attach the original workspace. Source and binaries must identify the same reviewed commit.

1. Review the release report and remaining acceptance gates. Push release-v1.0.2 with your existing Git login; open a PR into main using the existing GitHub CLI or website. Wait for Windows release checks and merge.
2. Create the new annotated v1.0.2 tag on the exact packaged commit and push it. Do not retag an already published version.
3. Create a new Release draft on v1.0.2. Use docs/releases/v1.0.2.md as the English body; the Chinese link names the exact tag.
4. Upload the exact files listed by final RELEASE_ASSETS.json plus SHA256SUMS.txt. Include DIVA-Setup-v1.0.2.exe, three DIVA application volumes, four unchanged optional offline model files, notices/licenses, the manifest and bilingual notes. Never upload private .state, build inputs or the unsplit internal ZIP.
5. Verify the uploaded asset names, sizes and digests before publishing. Keep the Release a draft until all pre-publication gates pass.
6. For public online installation, the repo and release assets must be anonymously accessible. Repository visibility and source license are owner decisions, not changed by build scripts. No tokens are embedded.
7. Publish as a normal release only after authorization; then verify anonymous asset access. Failure of that check means the hosted installation gate remains unpassed. Do not hide it by clearing prerelease labels.

This candidate is unsigned unless the final report explicitly states trusted signing was verified. It does not promise certified original-singing accuracy. Old release/model assets remain unchanged.
