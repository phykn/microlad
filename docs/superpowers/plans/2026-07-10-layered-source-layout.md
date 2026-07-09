# Layered Source Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Group the thirteen current source packages under `common`, `modeling`, `pipelines`, and `app` without changing behavior or retaining old import paths.

**Architecture:** Preserve every current responsibility package as a child package and add one ownership layer above it. Imports follow `app -> pipelines -> modeling -> common`; tests mirror the same ownership while the cross-layer mathematical audit remains at `tests/math_audit`.

**Tech Stack:** Python, PyTorch, NumPy, pytest, Git, Jupyter notebook JSON

## Global Constraints

- Keep function names, signatures, values, mathematical behavior, outputs, and exceptions unchanged.
- Do not retain compatibility modules or re-exports for the former top-level imports.
- Keep `tests/math_audit` at the test root because it validates multiple layers.
- Update maintained scripts, README examples, audit documentation, and notebook imports.
- Finish with exactly `common`, `modeling`, `pipelines`, and `app` as directories directly under `src`.

---

### Task 1: Add source ownership layers

**Files:**
- Create: `src/common/__init__.py`
- Create: `src/modeling/__init__.py`
- Create: `src/pipelines/__init__.py`
- Create: `src/app/__init__.py`
- Move: `src/helpers` to `src/common/helpers`
- Move: `src/neural` to `src/common/neural`
- Move: `src/tensors` to `src/common/tensors`
- Move: `src/phases` to `src/modeling/phases`
- Move: `src/vae` to `src/modeling/vae`
- Move: `src/diffusion` to `src/modeling/diffusion`
- Move: `src/data` to `src/pipelines/data`
- Move: `src/training` to `src/pipelines/training`
- Move: `src/reconstruction` to `src/pipelines/reconstruction`
- Move: `src/guidance` to `src/pipelines/guidance`
- Move: `src/scaling` to `src/pipelines/scaling`
- Move: `src/api` to `src/app/api`
- Move: `src/runtime` to `src/app/runtime`
- Modify: every Python import under `src`, `tests`, `run_train_vae.py`, and `run_train_diffusion.py`

**Interfaces:**
- Produces: `src.app.api` for `AnchorSlice`, `PredictOptions`, and `Predictor`
- Produces: `src.app.runtime` for configuration, factories, loading, and `load_predictor`
- Produces: full new paths under `src.common`, `src.modeling`, and `src.pipelines`

- [ ] **Step 1: Verify the pre-refactor baseline**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
D:\code\microlad\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Expected: `392 passed, 72 subtests passed`.

- [ ] **Step 2: Create group packages and move source directories**

Create the four group `__init__.py` files as empty package markers. Use `git mv` for each directory in the exact mapping listed in this task.

- [ ] **Step 3: Verify stale imports fail after the move**

Run:

```powershell
D:\code\microlad\.venv\Scripts\python.exe -m pytest --collect-only -q -p no:cacheprovider
```

Expected: collection fails because imports such as `src.api`, `src.diffusion`, and `src.phases` no longer exist.

- [ ] **Step 4: Rewrite source and test imports with the complete ownership map**

Apply these exact prefix replacements to Python files under `src` and `tests`, plus both maintained training scripts:

```python
REPLACEMENTS = {
    "src.helpers": "src.common.helpers",
    "src.neural": "src.common.neural",
    "src.tensors": "src.common.tensors",
    "src.phases": "src.modeling.phases",
    "src.vae": "src.modeling.vae",
    "src.diffusion": "src.modeling.diffusion",
    "src.data": "src.pipelines.data",
    "src.training": "src.pipelines.training",
    "src.reconstruction": "src.pipelines.reconstruction",
    "src.guidance": "src.pipelines.guidance",
    "src.scaling": "src.pipelines.scaling",
    "src.api": "src.app.api",
    "src.runtime": "src.app.runtime",
}
```

This replacement also updates string-based patch targets such as `"src.runtime.torch..."` in tests. Do not add aliases for the old paths.

- [ ] **Step 5: Verify source imports and layer direction**

Run:

```powershell
D:\code\microlad\.venv\Scripts\python.exe -c "from src.app.api import Predictor; from src.app.runtime import load_predictor; from src.modeling.diffusion import DDPMProcess"
D:\code\microlad\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Expected: imports succeed and the full suite passes.

Inspect every `from src.<group>` and `import src.<group>` statement. Assign ranks `common=0`, `modeling=1`, `pipelines=2`, and `app=3`; every imported group rank must be less than or equal to the importing file's group rank.

- [ ] **Step 6: Commit source ownership**

```powershell
git add src tests run_train_vae.py run_train_diffusion.py
git commit -m "refactor: add source ownership layers"
```

### Task 2: Mirror ownership in tests

**Files:**
- Create: `tests/common/__init__.py`
- Create: `tests/modeling/__init__.py`
- Create: `tests/pipelines/__init__.py`
- Create: `tests/app/__init__.py`
- Move: `tests/helpers` to `tests/common/helpers`
- Move: `tests/tensors` to `tests/common/tensors`
- Move: `tests/phases` to `tests/modeling/phases`
- Move: `tests/vae` to `tests/modeling/vae`
- Move: `tests/diffusion` to `tests/modeling/diffusion`
- Move: `tests/data` to `tests/pipelines/data`
- Move: `tests/training` to `tests/pipelines/training`
- Move: `tests/reconstruction` to `tests/pipelines/reconstruction`
- Move: `tests/guidance` to `tests/pipelines/guidance`
- Move: `tests/scaling` to `tests/pipelines/scaling`
- Move: `tests/api` to `tests/app/api`
- Move: `tests/runtime` to `tests/app/runtime`
- Preserve: `tests/math_audit`

**Interfaces:**
- Consumes: new source imports produced by Task 1
- Produces: a test tree that mirrors source ownership without changing test behavior

- [ ] **Step 1: Create test group packages and move test directories**

Create the four test group `__init__.py` files as empty package markers. Use `git mv` for every test directory in the exact mapping above; do not move `tests/math_audit`.

- [ ] **Step 2: Verify collection and behavior**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
D:\code\microlad\.venv\Scripts\python.exe -m pytest --collect-only -q -p no:cacheprovider
D:\code\microlad\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Expected: collection succeeds and the same 392 tests plus 72 subtests pass.

- [ ] **Step 3: Verify the test tree mirrors source ownership**

Confirm that the direct test directories are `app`, `common`, `math_audit`, `modeling`, and `pipelines`, with no former responsibility directories left at the test root.

- [ ] **Step 4: Commit the test layout**

```powershell
git add tests
git commit -m "refactor: mirror source layers in tests"
```

### Task 3: Update maintained documentation and examples

**Files:**
- Modify: `README.md`
- Modify: `docs/audit/mathematical-audit.md`
- Modify: `notebooks/00_dataset.ipynb`
- Modify: `notebooks/01_vae.ipynb`
- Modify: `notebooks/02_diffusion.ipynb`
- Modify: `notebooks/03_predict.ipynb`
- Modify: `notebooks/04_scale_up.ipynb`

**Interfaces:**
- Consumes: public paths from Task 1
- Produces: maintained user-facing examples that import only from the new package hierarchy

- [ ] **Step 1: Replace documented package paths**

Apply the same `REPLACEMENTS` map from Task 1 to the README, mathematical audit, and notebook JSON text. Update the README source-tree section to describe the four ownership layers and use `src.app.runtime.load_predictor` as the maintained loading path.

- [ ] **Step 2: Search maintained surfaces for stale imports**

Run:

```powershell
rg -n "src\.(helpers|neural|tensors|phases|vae|diffusion|data|training|reconstruction|guidance|scaling|api|runtime)(\.|\b)" src tests run_train_vae.py run_train_diffusion.py README.md notebooks docs/audit
```

Expected: no matches. Historical design and plan documents under `docs/superpowers` are intentionally excluded.

- [ ] **Step 3: Run final verification**

Run the public imports, layer-direction check, source/test directory checks, and complete test suite again. Expected results:

- only `app`, `common`, `modeling`, and `pipelines` are direct source directories;
- only `app`, `common`, `math_audit`, `modeling`, and `pipelines` are direct test directories;
- no lower layer imports from a higher layer;
- `392 passed, 72 subtests passed`;
- `git diff --check` reports no errors.

- [ ] **Step 4: Commit maintained path updates**

```powershell
git add README.md docs/audit notebooks
git commit -m "docs: update layered package paths"
```
