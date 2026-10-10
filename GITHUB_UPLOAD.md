# Publishing pre-1.0.1a

This source checkout uses the existing repository history. The development branch
is release-pre-1.0.1a, based on the published pre-1.0.0 commit. Do not recreate the
repository, force-push main, or rewrite the old tag.

## Source and main

Review the branch diff and Windows checks, then commit/push this branch and open a
pull request into main. Merging the PR brings the English README and README_CN.md
to the default branch. Keep repository visibility and the application source
license as separate owner decisions.

~~~text
git status --short
git diff --check
git push -u origin release-pre-1.0.1a
gh pr create --base main --head release-pre-1.0.1a --title "Prepare pre-1.0.1a reliability release" --body-file <prepared-PR-description>
~~~

The commands above are maintainer instructions; preparing these files does not
push or publish anything.

## Release

After the reviewed commit is present remotely, create a new tag named
pre-1.0.1a on that commit. Mark the Release as a pre-release. Use
docs/releases/pre-1.0.1a.md as the English body; its Chinese link points to this
exact tag.

Upload all 15 files from the separately verified
Any2Xsampa-Release-assets-pre-1.0.1a directory. They include Setup, three
application volumes, four optional offline model files, model notices/licenses,
the split manifest, both release descriptions and SHA256SUMS.txt.

Do not attach the unsplit internal ZIP, private build inputs, study results,
audio, credentials or the original workspace. Do not upload this maintainer
guide as a release asset. Public checksums cover the other 14 attachments and
exclude the checksum file itself.

Automatic model installation uses the same fixed upstream sources as pre-1.0.0.
The four offline model attachments are byte-identical and reusable. Install the
application into a new directory; the installer preserves old installations.

This build is unsigned. Signing and publication are distinct actions, not
implied by a successful local build. Do not silently replace old release assets.
See docs/release_build.md for the signing sequence and verification limits.
