# Releasing the `trustmark` crate

The Rust crate in this directory is published to
[crates.io](https://crates.io/crates/trustmark) independently of the Python
package at the repository root. This document describes how that happens.

## What gets published

Only the root package of the `rust/` workspace — `trustmark` — is published.
The workspace members under `rust/crates/` (`trustmark-cli` and `xtask`) are
marked `publish = false` and are intentionally not released to crates.io.

The ONNX model files are **not** part of the published crate. Consumers fetch
them separately (see [`README.md`](README.md)); `rust/models/` is gitignored
apart from a `.gitkeep`.

## Automated release (preferred)

1. Update `version` in `rust/Cargo.toml`.
2. Run `cargo check` (or any cargo command) so `rust/Cargo.lock` picks up the
   new version, and add a section to [`CHANGELOG.md`](CHANGELOG.md).
3. Land those changes on `main` through a pull request. The
   [Rust CI](../.github/workflows/rust-ci.yml) workflow must be green.
4. Tag the merge commit and push the tag:

   ```sh
   git tag rust-v0.3.0
   git push origin rust-v0.3.0
   ```

   The `rust-v` prefix is required; it keeps crate releases distinguishable
   from Python package tags in the same repository.

Pushing the tag runs [Rust release](../.github/workflows/rust-release.yml),
which verifies that the tag matches `rust/Cargo.toml`, confirms the version
isn't already on crates.io, runs the tests, packages the crate, publishes it,
and opens a GitHub release.

To rehearse without publishing, run the workflow manually from the Actions tab
with `dry_run` left enabled.

### One-time crates.io setup

The release workflow authenticates with
[Trusted Publishing](https://crates.io/docs/trusted-publishing), so no API
token is stored in this repository. An owner of the `trustmark` crate must
configure it once, under **Settings → Trusted Publishing** on the crate page:

| Field          | Value                |
| -------------- | -------------------- |
| Repository     | `adobe/trustmark`    |
| Workflow file  | `rust-release.yml`   |
| Environment    | `crates-io`          |

If you would rather use a long-lived token, add it as a repository secret
named `CARGO_REGISTRY_TOKEN` and replace the "Authenticate to crates.io" step
in the release workflow with that secret. Trusted Publishing is preferred
because the credential is short-lived and scoped to this one workflow.

## Manual release

Occasionally you may need to publish from a workstation — for example, when
restoring a crate whose automation is not yet in place.

Prerequisites: a crates.io account that is an owner of the `trustmark` crate,
and `cargo login` already run (or `CARGO_REGISTRY_TOKEN` exported).

From this directory (`rust/`):

```sh
# 1. Confirm the version you are about to publish.
grep -m1 '^version' Cargo.toml

# 2. Fetch the models, which the tests need.
cargo xtask fetch-models

# 3. Validate.
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features
cargo test --workspace --locked
cargo publish --dry-run --locked

# 4. Publish.
cargo publish --locked

# 5. Record the release in git so the history is not lost.
git tag rust-v$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)
git push origin --tags
```

Always push the tag. Releases 0.1.0 through 0.2.2 were published by hand
without tags, which made it hard to tell what source a given crates.io version
corresponded to.

## Dependency risk: `ort` release candidates

The crate depends on [`ort`](https://ort.pyke.io) with an exact version pin
(`=2.0.0-rc.N`). `ort-sys` downloads prebuilt ONNX Runtime binaries from a
pyke-hosted CDN at build time, and **those binaries are removed for older
release candidates**. When that happens, an already-published version of
`trustmark` stops building even though nothing on crates.io was yanked.

This is exactly what happened to `trustmark` 0.2.2, which pins
`ort =2.0.0-rc.8`. The fix is to upgrade the pin and publish a new version;
there is no way to repair a published version in place.

Practical guidance:

- Treat an `ort` release-candidate upgrade as release-worthy on its own.
- When `ort` publishes a new rc, upgrade reasonably promptly rather than
  waiting for an unrelated feature to justify a release.
- Keep the [platform support section](README.md#platform-support) of the
  README in sync with whichever rc is pinned.
