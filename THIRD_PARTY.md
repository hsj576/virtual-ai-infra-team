# Third-party models and dependencies

Virtual AI Infra Team is licensed under Apache-2.0. Model weights and Python dependencies retain their own licenses and are not redistributed in this repository.

## Models used by the v0.1 reference path

| Component | Repository | Fixed revision | License status |
|---|---|---|---|
| Target | `mlx-community/Qwen3.8-27B-4bit` | `3e6447f082e89cc7f0bc6e5441afd38dfce760ff` | Review the model repository before download |
| DFlash2 drafter | `z-lab/Qwen3.8-27B-DFlash2` | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` | Manifest-reviewed as Apache-2.0 |
| Optional fallback planner | `mlx-community/Qwen3.5-4B-MLX-4bit` | Not enabled by default | Review before enabling |

The application downloads model assets directly from their providers. Users are responsible for complying with the applicable model terms.

## Runtime dependencies

The release pins MLX, MLX-VLM, and MLX-LM to preserve benchmark comparability. See `pyproject.toml` for exact versions. Other direct dependencies include `huggingface_hub` and `PyYAML`.

This file is informational and not legal advice.
