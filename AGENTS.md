## User Wiki

For non-trivial work, first read `C:\Users\KN\.codex\user-wiki\AGENTS.md` and follow it.
Current user requests, system/developer instructions, and this file take precedence over the user wiki.

## Repository Shape

Active code lives in `src`.

Do not depend on removed legacy folders such as `src_backup`, `reference`, `microlad-anode`, `output`, or `notebooks`.

Maintained entrypoints:

- VAE training: `run_train_vae.py`
- Diffusion training: `run_train_diffusion.py`
- Configs: `config/vae.yaml`, `config/diffusion.yaml`
- Prediction loading: `src.build.load_predictor`

Generated training output goes under `run/<timestamp>` and is not source code.

## Paper Graph

The paper graph lives in `docs/paper/graph`.

If editing it:

- Write Korean.
- Use full vault-path wikilinks like `[[docs/paper/graph/claim|논문의 주장]]`.
- Verify graph links, markdown links, image links, and local reference ids before finishing.
