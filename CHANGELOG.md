# Changelog

## [Unreleased]

### Fixed
- Stop silent loss of multiple boxes; honor generator df/path arguments.
- Replace misleading coordinate categorical accuracy with regression MAE; disable upload/unzip and clear notebook output.

### Added
- Strict single-box/group CSV, image/coordinate checks, RGB/fractional-label pipeline and explicit CPU training CLI.
- Actual reduced-stack synthetic backend checks, uv lock, Ruff/mypy/hooks and data-only Docker/CI.
- Honest single-box/large-default/model-quality limits, no mAP claim.


### Engineering follow-up
- Use the committed uv lock in CI and key dependency-installer caching by that lock.
- Complete copy-only profile README drafts and clarify historical versus current verification.

### Synthetic demonstration follow-up

Added standalone invented-input backend demo, checked JSON output and documented limits. Opt-in ML container added where a model backend exists; default lightweight image unchanged. Hosted container verification is required separately.

## Portable notebook follow-up

Replace platform-specific notebook execution with safe-default local package entry points, explicit local input checks and opt-in training. Add notebook execution/argv/failure regressions. Historical model scores and full-scale data training are not claimed reproduced.
