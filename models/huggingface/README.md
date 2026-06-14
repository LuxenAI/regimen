# Hugging Face Specialist Models

This folder is for the Hugging Face developer-agent subroutine checkpoints used by
`slm-harness`.

The production code can use these models by setting `.slm-harness.toml` routes to
`local_transformers` and pointing `model_path` at the matching folder under
`models/huggingface/`.

The source-of-truth manifest is `manifest.json`. Full checkpoints are stored with Git LFS
when vendored into this repository.
