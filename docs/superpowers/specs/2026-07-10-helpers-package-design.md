# Helpers package design

## Goal

Group small, reusable image-loading and image-segmentation utilities without creating
separate top-level packages for each concern.

## Structure

Create a `src.helpers` package with two focused modules:

- `src/helpers/images.py` owns image-file loading and conversion to grayscale or phase labels.
- `src/helpers/segmentation.py` owns conversion of grayscale intensity images to discrete phase labels.

Remove `src/io` and `src/phases/segmentation.py`. Keep the remaining phase representation,
relaxation, and quantization code in `src/phases` because those modules implement phase-domain
mathematics rather than input helpers.

Mirror this structure under `tests/helpers`. Update every production and test import to use
`src.helpers`; no compatibility re-exports will remain.

## Behavior and errors

This is a structural change only. Function names, accepted inputs, returned dtypes, and error
behavior remain unchanged. In particular, segmentation continues to use multi-level Otsu
thresholds and returns `uint8` phase labels.

## Verification

Run the helper, data, guidance, and full test suites. Search the repository for imports from
`src.io` and `src.phases.segmentation`, then compile `src` to detect stale module references.
