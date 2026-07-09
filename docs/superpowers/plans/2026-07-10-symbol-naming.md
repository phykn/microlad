# Symbol Naming and Helper Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename unclear or unnecessarily long symbols and reduce duplicated or one-use helpers without changing valid model calculations.

**Architecture:** Introduce two shared scalar validators, one shared VAE geometry function, and two shared latent decoders. Update definitions and every caller directly; do not retain aliases for old names. Inline only helpers whose call-site expression is clearer than the extra definition.

**Tech Stack:** Python, PyTorch, NumPy, pytest, AST, Jupyter notebook JSON, Git

## Global Constraints

- Functions and methods use concise operation verbs; classes use concise noun phrases.
- Preserve standard mathematical and PyTorch names such as `forward`, `q_sample`, `p_sample`, `kl_divergence`, and `*_loss`.
- Preserve valid-input tensor values, shapes, gradients, and model behavior.
- Do not retain compatibility aliases for renamed or removed symbols.
- Update source, tests, README, notebooks, exports, patch targets, and maintained docs.
- Work directly on `main`, which is the user's requested final branch state.

---

### Task 1: Consolidate scalar validation

**Files:**
- Create: `src/common/validation.py`
- Create: `tests/common/test_validation.py`
- Modify: `src/app/api/options.py`
- Modify: `src/app/api/preparation.py`
- Modify: `src/pipelines/guidance/conditioning/targets.py`
- Modify: `src/pipelines/reconstruction/slices.py`
- Modify: `src/pipelines/reconstruction/volume.py`
- Modify: `src/pipelines/scaling/tiles.py`

**Interfaces:**
- Produces: `require_int(name: str, value: int) -> None`
- Produces: `require_finite_number(name: str, value: float) -> None`

- [ ] **Step 1: Add tests for the new shared validators**

Create tests that accept ordinary integers and finite real scalars, reject booleans as integers or real scalars, reject non-integers, and reject `nan` and infinities. Import only:

```python
from src.common.validation import require_finite_number, require_int
```

- [ ] **Step 2: Run the new test and verify the module is missing**

Run: `D:\code\microlad\.venv\Scripts\python.exe -m pytest tests/common/test_validation.py -q -p no:cacheprovider`

Expected: collection fails with `ModuleNotFoundError: No module named 'src.common.validation'`.

- [ ] **Step 3: Implement the validators**

```python
import math
from numbers import Real


def require_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")


def require_finite_number(name: str, value: float) -> None:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a real scalar.")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite.")
```

- [ ] **Step 4: Replace duplicates**

Import `require_int` in all six consumers and remove their `_validate_integer` definitions. Import `require_finite_number` in options and guidance targets, replace both `_validate_finite_scalar` call sites, and remove both duplicate definitions.

- [ ] **Step 5: Verify focused tests and commit**

Run:

```powershell
D:\code\microlad\.venv\Scripts\python.exe -m pytest tests/common tests/app/api tests/pipelines/guidance/test_conditioning_targets.py tests/pipelines/reconstruction tests/pipelines/scaling/test_tiles.py -q -p no:cacheprovider
```

Expected: all selected tests pass.

Commit: `refactor: share scalar validation`

### Task 2: Consolidate VAE geometry and latent decoding

**Files:**
- Modify: `src/modeling/vae/model.py`
- Modify: `src/modeling/vae/__init__.py`
- Modify: `src/app/api/predictor.py`
- Modify: `src/app/api/preparation.py`
- Modify: `src/pipelines/guidance/conditioning/latents.py`
- Modify: `src/pipelines/scaling/conditioning.py`
- Modify: `src/pipelines/scaling/decoding.py`
- Modify: `src/pipelines/reconstruction/volume.py`
- Modify: `src/pipelines/reconstruction/__init__.py`
- Modify: `src/pipelines/guidance/evaluation.py`
- Modify: `src/pipelines/guidance/optimization.py`
- Modify: `src/pipelines/scaling/local_objective.py`
- Modify: `tests/modeling/vae/test_model.py`
- Modify: `tests/pipelines/reconstruction/test_volume.py`

**Interfaces:**
- Produces: `get_downsample_factor(vae: torch.nn.Module) -> int`
- Produces: `decode_latent(vae: torch.nn.Module, latent: torch.Tensor) -> torch.Tensor`
- Produces: `decode_latents(vae: torch.nn.Module, latents: torch.Tensor) -> torch.Tensor`

- [ ] **Step 1: Add failing tests for shared geometry and decoders**

Test that `get_downsample_factor` returns a valid explicit factor, derives a missing factor from image and latent sizes, and rejects non-positive or inconsistent geometry. Test that `decode_latent` returns `[H, W]`, `decode_latents` returns `[B, H, W]`, and both reject invalid decoder shapes or non-finite output.

- [ ] **Step 2: Run focused tests and verify the new exports are missing**

Run the selected VAE and reconstruction tests. Expected: imports of the three new names fail.

- [ ] **Step 3: Implement and export VAE geometry**

Implement `get_downsample_factor` in `src/modeling/vae/model.py` using the existing strict positive and geometry consistency checks. Export it from `src/modeling/vae/__init__.py`. Replace all repeated `_downsample_factor` functions and the Predictor method with direct calls to this function.

- [ ] **Step 4: Implement and export latent decoders**

Move the duplicated single and batch decoder implementations into `src/pipelines/reconstruction/volume.py` as `decode_latent` and `decode_latents`. Export them from the reconstruction package, update guidance and scaling imports, then remove all four duplicate private definitions.

- [ ] **Step 5: Verify focused and full tests, then commit**

Run the VAE, reconstruction, guidance, scaling, and API suites, followed by the complete suite. Expected: `392 passed, 72 subtests passed` plus the newly added validator/geometry tests.

Commit: `refactor: share vae geometry helpers`

### Task 3: Remove one-use indirection

**Files:**
- Modify: `src/app/runtime/factories.py`
- Modify: `src/modeling/diffusion/sampler.py`
- Modify: `src/pipelines/guidance/physics/diffusivity.py`
- Modify: `src/pipelines/guidance/conditioning/targets.py`
- Modify: `src/pipelines/guidance/conditioning/reconstruction.py`
- Modify: `src/pipelines/reconstruction/refinement.py`
- Modify: `src/pipelines/training/runtime.py`

**Interfaces:**
- Consumes: `tile_grid` from `src.pipelines.scaling.tiles`
- Produces: no new public interface

- [ ] **Step 1: Inline the simple helpers**

Apply these exact changes:

- build the image path comprehension directly inside `build_dataset` and remove `_image_paths_from_dir`;
- place the cubic L-MPDD condition directly after `_validate_shape` and remove `_validate_lmpdd_shape`;
- place the unit-interval condition at its only call and remove `_validate_unit_interval`;
- place the target-weight predicate at its only call and remove `_uses_any_target`;
- validate `steps` at the start of `refine_axes` and remove `_validate_steps`;
- call `datetime.now().strftime(...)` at run-directory construction and remove `_timestamp`;
- import maintained `tile_grid` in anchor reconstruction and remove its local `_tile_grid` and `_tile_starts`.

- [ ] **Step 2: Verify removed definitions and behavior**

Search for all eight removed definitions and references. Expected: no matches except the maintained `tile_grid` definition. Run diffusion sampler, guidance, reconstruction, runtime, and scaling tile tests.

- [ ] **Step 3: Commit**

Commit: `refactor: remove one-use helpers`

### Task 4: Rename application, common, and modeling symbols

**Files:**
- Modify: all affected files under `src/app`, `src/common`, and `src/modeling`
- Modify: corresponding files under `tests/app`, `tests/common`, `tests/modeling`, and `tests/math_audit`
- Modify: `run_train_vae.py`
- Modify: `run_train_diffusion.py`

**Interfaces:**
- Produces: the class and public runtime/common/modeling names listed in the design specification
- Removes: every corresponding old name without an alias

- [ ] **Step 1: Change focused tests to import and patch the new names**

Update tests for `PredictionPrep`, `TimeResBlock`, `TimeResStack`, `load_defaults`, `apply_vae_defaults`, `build_denoiser`, `build_ddpm`, `load_run_vae`, `load_denoiser`, `build_predictor`, `require_float`, `require_finite`, `segment_otsu`, `decode_phase`, `calc_phase_probs`, `decode_probs`, and `decode_relaxed`.

- [ ] **Step 2: Verify old source fails the new imports**

Run app, common, modeling, and mathematical audit test collection. Expected: import or attribute failures for the new names.

- [ ] **Step 3: Rename definitions, exports, call sites, and patch strings**

Apply the exact class, public symbol, and application/common/modeling private rename tables from `docs/superpowers/specs/2026-07-10-symbol-naming-design.md`. Use identifier-boundary replacement and inspect the diff so substrings inside unrelated identifiers are unchanged.

- [ ] **Step 4: Verify and commit**

Run app, common, modeling, runtime-entrypoint, and mathematical audit tests. Search those maintained surfaces for old names. Commit: `refactor: clarify core symbol names`.

### Task 5: Rename pipeline and training symbols

**Files:**
- Modify: all affected files under `src/pipelines`
- Modify: corresponding files under `tests/pipelines`, `tests/app`, and `tests/math_audit`

**Interfaces:**
- Produces: the public guidance, reconstruction, scaling, training, and private names listed in the design specification
- Removes: every corresponding old name without an alias

- [ ] **Step 1: Change focused tests to new pipeline names**

Update tests and patch strings for `encode_anchors`, `reconstruct_target`, `build_anchor_targets`, `freeze_inference`, `build_phase_target`, `sample_descriptor_loss`, `refine_axes`, `shift_anchor_slices`, `encode_scale_anchors`, `build_scale_targets`, `decode_large_volume`, `normalize_tile_weights`, `validate_training`, `format_progress`, `write_checkpoint`, `replace_atomic`, `calc_grad_norm`, and `unpack_batch`.

- [ ] **Step 2: Verify new imports fail before implementation**

Run pipeline tests. Expected: import or attribute failures for the renamed names.

- [ ] **Step 3: Apply the exact pipeline and private rename maps**

Rename definitions, imports, exports, method calls, and string patch targets exactly as specified in the design. Do not rename mathematical `*_loss` functions or framework methods.

- [ ] **Step 4: Verify and commit**

Run pipeline, API, and mathematical audit tests. Search source and tests for every old pipeline symbol. Commit: `refactor: clarify pipeline symbol names`.

### Task 6: Update maintained examples and audit the whole symbol table

**Files:**
- Modify: `README.md`
- Modify: `docs/audit/mathematical-audit.md`
- Modify: `notebooks/00_dataset.ipynb`
- Modify: `notebooks/01_vae.ipynb`
- Modify: `notebooks/02_diffusion.ipynb`
- Modify: `notebooks/03_predict.ipynb`
- Modify: `notebooks/04_scale_up.ipynb`

**Interfaces:**
- Consumes: all new names from Tasks 1 through 5
- Produces: maintained examples with no stale symbol reference

- [ ] **Step 1: Update README, audit, and notebook references**

Apply only identifier changes from the approved specification. Parse every notebook as JSON after replacement.

- [ ] **Step 2: Run AST and stale-symbol audits**

Parse every Python file with `ast.parse`. Search source, tests, scripts, README, notebooks, and `docs/audit` for every old name and fail on any match. Recount definitions and report the helper reduction from the original 341 definitions.

- [ ] **Step 3: Run complete verification**

Run public import smoke checks, source-layer direction validation, notebook JSON parsing, and the complete test suite with Python bytecode and pytest cache disabled. Run `git diff --check` and confirm the worktree is clean after commit.

- [ ] **Step 4: Commit**

Commit: `docs: update renamed symbols`
