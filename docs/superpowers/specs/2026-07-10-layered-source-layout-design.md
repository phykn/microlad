# Layered source layout design

## Goal

Reduce the number of immediate children under `src` while preserving the current
responsibility-focused packages and their behavior.

## Package layout

The source tree will use four top-level groups:

```text
src/
  common/
    helpers/
    neural/
    tensors/

  modeling/
    phases/
    vae/
    diffusion/

  pipelines/
    data/
    training/
    reconstruction/
    guidance/
    scaling/

  app/
    api/
    runtime/
```

Each group owns one mutually exclusive responsibility:

- `common` contains reusable low-level operations without workflow ownership.
- `modeling` contains mathematical representations and learned generative models.
- `pipelines` contains data, training, reconstruction, guidance, and scale-up workflows.
- `app` contains public application entrypoints and runtime object construction.

Each group and existing child package remains a Python package with an `__init__.py`.
No child package will be flattened or merged during this change.

## Dependency direction

The intended direction is:

```text
app -> pipelines -> modeling -> common
```

A package may depend on another package in the same group when their current domain
relationship requires it. Lower groups must not import from a higher group. Existing
same-group relationships, including guidance and scaling collaboration, remain unchanged.

## Public imports and compatibility

All imports will use their full new ownership path. Examples:

```python
from src.app.runtime import load_predictor
from src.app.api import AnchorSlice, PredictOptions
from src.modeling.diffusion import DDPMProcess
from src.pipelines.scaling.optimization import optimize_large_volume
from src.common.helpers.images import load_image
```

Compatibility re-exports for old paths such as `src.runtime`, `src.diffusion`, or
`src.guidance` will not be retained. Function names, signatures, accepted values,
mathematical behavior, returned values, and exceptions remain unchanged.

Maintained scripts, README examples, configs, and notebook import strings will be updated
to use the new paths.

## Tests

Tests will mirror source ownership:

```text
tests/
  common/
    helpers/
    tensors/
  modeling/
    phases/
    vae/
    diffusion/
  pipelines/
    data/
    training/
    reconstruction/
    guidance/
    scaling/
  app/
    api/
    runtime/
  math_audit/
```

`math_audit` remains top-level because it verifies equations and invariants across multiple
layers. Test implementations change only where import paths or test locations must follow
the new source ownership.

## Verification

Verification will include:

1. Search all maintained files for imports from the former top-level package paths.
2. Check that only `common`, `modeling`, `pipelines`, and `app` remain as source directories.
3. Import the maintained training and prediction entrypoints.
4. Run the complete test suite without preserving compatibility modules.
5. Confirm the Git worktree is clean after the structural commit.
