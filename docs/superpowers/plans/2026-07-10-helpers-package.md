# Helpers Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move image loading and grayscale-to-phase segmentation into a focused `src.helpers` package without preserving legacy imports or changing behavior.

**Architecture:** `src.helpers.images` owns file decoding and dtype conversion, while `src.helpers.segmentation` owns Multi-Otsu phase labeling. Consumers import these helpers directly; phase-domain representation and optimization remain in `src.phases`.

**Tech Stack:** Python, NumPy, Pillow, scikit-image, pytest, Git

## Global Constraints

- Do not preserve compatibility imports for `src.io` or `src.phases.segmentation`.
- Preserve all function signatures, return dtypes, validation, and error behavior.
- Keep image loading and segmentation in separate modules under `src.helpers`.
- Mirror production ownership under `tests/helpers`.

---

### Task 1: Establish the helpers package boundary

**Files:**
- Create: `tests/helpers/test_images.py`
- Create: `tests/helpers/test_segmentation.py`
- Delete: `tests/io/test_images.py`
- Delete: `tests/phases/test_segmentation.py`
- Create: `src/helpers/__init__.py`
- Create: `src/helpers/images.py`
- Create: `src/helpers/segmentation.py`
- Delete: `src/io/__init__.py`
- Delete: `src/io/images.py`
- Delete: `src/phases/segmentation.py`
- Modify: `src/phases/__init__.py`

**Interfaces:**
- Produces: `load_image(path: str | Path) -> np.ndarray`
- Produces: `load_phase_image(path: str | Path) -> np.ndarray`
- Produces: `segment_multi_otsu(image: np.ndarray, num_phases: int) -> np.ndarray`

- [ ] **Step 1: Move tests to the new ownership and update imports**

Move `tests/io/test_images.py` to `tests/helpers/test_images.py` and change its import to:

```python
from src.helpers.images import load_image, load_phase_image
```

Move `tests/phases/test_segmentation.py` to `tests/helpers/test_segmentation.py` and change its import to:

```python
from src.helpers.segmentation import segment_multi_otsu
```

- [ ] **Step 2: Run the focused tests and verify the new package is missing**

Run: `python -m pytest tests/helpers/test_images.py tests/helpers/test_segmentation.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'src.helpers'`.

- [ ] **Step 3: Move implementations and define the package exports**

Move `src/io/images.py` to `src/helpers/images.py` without changing its implementation. Move `src/phases/segmentation.py` to `src/helpers/segmentation.py` without changing its implementation. Create `src/helpers/__init__.py` with:

```python
from src.helpers.images import load_image, load_phase_image
from src.helpers.segmentation import segment_multi_otsu

__all__ = ["load_image", "load_phase_image", "segment_multi_otsu"]
```

Remove the segmentation import and `segment_multi_otsu` export from `src/phases/__init__.py`, then remove the obsolete `src/io` package.

- [ ] **Step 4: Run the focused tests**

Run: `python -m pytest tests/helpers/test_images.py tests/helpers/test_segmentation.py -q`

Expected: all helper tests pass.

### Task 2: Migrate consumers and verify repository integrity

**Files:**
- Modify: `src/data/dataset.py`
- Modify: `src/guidance/conditioning/images.py`
- Modify: `src/guidance/conditioning/targets.py`

**Interfaces:**
- Consumes: the three exports from `src.helpers`
- Produces: no new interface; all existing dataset and guidance behavior remains unchanged

- [ ] **Step 1: Replace legacy imports**

Use these direct imports:

```python
from src.helpers.images import load_image, load_phase_image
from src.helpers.segmentation import segment_multi_otsu
```

Import only the functions used by each module.

- [ ] **Step 2: Verify no legacy import remains**

Run: `rg -n "src\\.io|src\\.phases\\.segmentation|from src\\.phases import segment_multi_otsu" src tests`

Expected: no matches.

- [ ] **Step 3: Run affected suites**

Run: `python -m pytest tests/helpers tests/data tests/guidance -q`

Expected: all affected tests pass.

- [ ] **Step 4: Run complete verification**

Run: `python -m compileall -q src`

Expected: exit code 0.

Run: `python -m pytest -q`

Expected: the complete suite passes.

- [ ] **Step 5: Commit the structural change**

```powershell
git add src/helpers src/data/dataset.py src/guidance/conditioning/images.py src/guidance/conditioning/targets.py src/phases/__init__.py tests/helpers
git add -u src/io src/phases/segmentation.py tests/io tests/phases/test_segmentation.py
git commit -m "refactor: group image helpers"
```
