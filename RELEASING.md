# Releasing Dendra Models

Update `src/dendra_models/__init__.py` and add a matching `## <version>` section
to `CHANGELOG.md` on `main` before tagging a release. The GitHub tag must be
exactly `v` followed by that package version, and its commit must belong to
`main`.

The **Build and Publish Distributions** workflow builds one universal wheel and
one source distribution, checks their metadata and packaged resources, and tests
installed copies on Linux and macOS. Only those tested files become the
`checked-dists` artifact used for publication.

A version tag pushed to `wmglab-duke/dendra-models` publishes `checked-dists` to
production PyPI through the `pypi` environment. After that upload succeeds, the
workflow creates a GitHub Release for the same tag with notes from `CHANGELOG.md`
and attaches those same distribution files. The release job alone receives
`contents: write`; publishing uses the existing PyPI trusted publisher.

Pull requests, pushes to `main`, and validation-only manual runs do not publish
to PyPI or create GitHub Releases. A manual run with `publish_target=testpypi`
publishes only to TestPyPI and does not create a GitHub Release.

## Recovering a failed release step

If PyPI publication succeeded but the GitHub Release job failed, rerun the
failed release job in that workflow run. Do not rerun the successful PyPI job:
PyPI does not allow replacing uploaded files.

The release job resumes a draft containing a verified subset of the expected
files, or verifies an existing published release. Matching assets are retained.
A partial upload, unexpected asset, or conflicting file fails the job and
requires manual review; the job does not overwrite or delete it. A published
release must already contain the complete matching asset set.

Keep the validated `checked-dists` artifact available for retries (the workflow
retains it for seven days). Investigate a tag, commit, or asset mismatch before
retrying; do not move the tag or replace released files.
