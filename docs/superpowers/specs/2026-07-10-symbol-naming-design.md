# Symbol naming and helper cleanup design

## Goal

Make function, method, and class names concise and responsibility-oriented across the
project, while removing helpers that add indirection or duplicate another implementation.

The current source contains 341 function, method, and class definitions, including 131
public module-level symbols. This change preserves valid-input calculations and model
behavior; it does not retain compatibility aliases for renamed symbols.

## Naming rules

- Functions and methods use a concise operation verb such as `apply`, `build`, `calc`,
  `decode`, `encode`, `format`, `load`, `require`, or `resolve`.
- Classes remain concise noun phrases.
- A module's responsibility is not repeated in every symbol when removing it stays clear.
- Names longer than 24 characters require review, but mathematical precision takes priority
  over a hard length limit.
- Established mathematical and framework conventions remain unchanged when they are already
  the clearest form: `forward`, `q_sample`, `p_sample`, `kl_divergence`, and `*_loss`.
- Standard domain abbreviations such as VAE, DDPM, SDS, TPC, and L-MPDD remain uppercase in
  class names and lowercase in Python identifiers.
- `require_*` raises when a scalar or dtype precondition is not satisfied. `validate_*`
  checks the structure or consistency of a compound object.

## Class renames

| Current | New |
| --- | --- |
| `PredictionPreparation` | `PredictionPrep` |
| `TimeResidualBlock` | `TimeResBlock` |
| `TimeResidualStack` | `TimeResStack` |

Other class names are already concise nouns that identify their role and remain unchanged.

## Public function and method renames

### Application runtime

| Current | New |
| --- | --- |
| `load_config_defaults` | `load_defaults` |
| `fill_diffusion_defaults_from_run` | `apply_vae_defaults` |
| `build_diffusion_model` | `build_denoiser` |
| `build_diffusion_process` | `build_ddpm` |
| `load_frozen_vae_from_run` | `load_run_vae` |
| `load_frozen_diffusion_model` | `load_denoiser` |
| `build_predictor_from_run` | `build_predictor` |

### Common and modeling

| Current | New |
| --- | --- |
| `validate_floating_dtype` | `require_float` |
| `validate_finite_tensor` | `require_finite` |
| `segment_multi_otsu` | `segment_otsu` |
| `model_output_to_phase` | `decode_phase` |
| `soft_phase_probability` | `calc_phase_probs` |
| `PatchVAE.decode_probabilities` | `PatchVAE.decode_probs` |
| `PatchVAE.decode_relaxed_labels` | `PatchVAE.decode_relaxed` |

### Guidance, reconstruction, and scaling

| Current | New |
| --- | --- |
| `prepare_anchor_latents` | `encode_anchors` |
| `reconstruct_anchor_target` | `reconstruct_target` |
| `prepare_anchor_targets` | `build_anchor_targets` |
| `prepare_inference_module` | `freeze_inference` |
| `phase_vector_target` | `build_phase_target` |
| `descriptor_loss_per_sample` | `sample_descriptor_loss` |
| `three_axis_refinement` | `refine_axes` |
| `shifted_anchor_slices` | `shift_anchor_slices` |
| `prepare_scale_anchor_latents` | `encode_scale_anchors` |
| `prepare_scale_anchor_targets` | `build_scale_targets` |
| `decode_large_latent_volume` | `decode_large_volume` |
| `normalized_tile_weights` | `normalize_tile_weights` |

### Training

| Current | New |
| --- | --- |
| `validate_train_settings` | `validate_training` |
| `progress_postfix` | `format_progress` |
| `save_checkpoint_file` | `write_checkpoint` |
| `replace_file` | `replace_atomic` |
| `model_grad_norm` | `calc_grad_norm` |
| `image_from_batch` | `unpack_batch` |
| `DiffusionTrainer.grad_norm` | `DiffusionTrainer.calc_grad_norm` |
| `VAETrainer.grad_norm` | `VAETrainer.calc_grad_norm` |

## Private symbol renames

| Current | New |
| --- | --- |
| `_last_model_path` | `_last_checkpoint` |
| `_require_config_value` | `_require_value` |
| `_require_config_values` | `_require_values` |
| `_yaml_safe_value` | `_to_yaml` |
| `_vae_model_from_args` | `_make_vae` |
| `_load_model_checkpoint` | `_load_state` |
| `_load_frozen_checkpoint` | `_load_frozen` |
| `_float_phase_labels_to_uint8` | `_float_labels_to_uint8` |
| `_validate_descriptor_inputs` | `_validate_descriptor` |
| `_validate_optimization_contract` | `_validate_contract` |
| `_validate_timestep_range` | `_validate_range` |
| `_local_prior_objective_batch` | `_batch_prior_loss` |
| `_decode_tiled_image_batch` | `_decode_tiles` |
| `_optimize_large_slice_batch` | `_optimize_batch` |
| `_validate_tile_batch_size` | `_validate_batch_size` |
| `_validate_non_negative_integer` | `_require_nonnegative_int` |
| `_validate_non_negative_scalar` | `_require_nonnegative` |
| `_validate_positive_scalar` | `_require_positive` |
| `_validate_positive_integer` | `_require_positive_int` |
| `_validate_anchor_tensor_map` | `_validate_anchor_map` |
| `_validate_anchor_tensor_key` | `_validate_anchor_key` |
| `_predict_volume_size` | `_resolve_volume_size` |
| `_anchor_volume_size` | `_get_anchor_size` |
| `_scale_anchor_latents` | `_build_scale_latents` |
| `_uses_scale_anchor` | `_has_scale_anchor` |
| `_scale_anchor_schedule` | `_build_anchor_schedule` |
| `_random_unused_index` | `_pick_unused_index` |
| `_scale_descriptor_tile_size` | `_resolve_tile_size` |
| `_target_image_size` | `_get_target_size` |
| `_scale_latent_size` | `_calc_latent_size` |
| `_scale_tile_overlap` | `_resolve_overlap` |
| `_scale_refine_overlap` | `_calc_refine_overlap` |
| `_sds_t_max` | `_resolve_t_max` |
| `_generate_volume` | `_generate_base` |
| `_generate_large_volume` | `_generate_large` |
| `_refine_volume` | `_refine` |
| `_sds_kwargs` | `_build_sds_args` |
| `_validate_predict_inputs` | `_validate_inputs` |
| `_image_size` | `_get_image_size` |

## Helper removal

The following one-use helpers add less clarity than their call-site expression and will be
inlined or replaced by an existing implementation:

- `_timestamp`
- `_validate_lmpdd_shape`
- `_validate_unit_interval`
- `_uses_any_target`
- `_validate_steps`
- `_image_paths_from_dir`
- the local `_tile_grid` and `_tile_starts` in anchor reconstruction; use the maintained
  scaling tile implementation instead

Larger one-use helpers remain when they isolate a mathematical equation, tensor layout,
validation boundary, or optimization phase.

## Duplicate helper consolidation

### Scalar validation

Create `src/common/validation.py` with:

- `require_int(name, value)` replacing six identical `_validate_integer` definitions
- `require_finite_number(name, value)` replacing two identical
  `_validate_finite_scalar` definitions

The tensor-specific `require_float` and `require_finite` remain under
`src/common/tensors/validation.py`.

### VAE geometry

Create and export `get_downsample_factor(vae)` from the VAE model package. It replaces the
repeated `_downsample_factor` functions and Predictor method. It validates that the factor
is positive and that `image_size == latent_size * downsample_factor`.

### Latent decoding

Add `decode_latent(vae, latent)` and `decode_latents(vae, latents)` to reconstruction volume
operations. They replace duplicated `_decode_latent` and `_decode_latent_batch`
implementations in guidance evaluation and scaling local objectives.

## Maintained surfaces

All definitions, imports, exports, call sites, string-based patch targets, tests, README
examples, and notebook imports will use the new symbols. Historical Superpowers design and
plan documents remain historical records and are excluded from stale-symbol checks.

## Verification

1. Establish the current full-suite baseline.
2. Change tests to require the new public names and shared helpers, and verify that the old
   source fails those tests before implementation.
3. Run focused tests after each responsibility group is renamed.
4. Parse every source file with Python AST and confirm that each removed old symbol has no
   definition or reference in maintained surfaces.
5. Recount definitions and report the net helper reduction.
6. Run public import smoke checks, notebook JSON parsing, layer-direction validation, and the
   complete test suite.
7. Confirm only `main` exists and the worktree is clean.
