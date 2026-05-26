# Data Directory

Store local datasets, symbolic links, degradation profiles, splits, and preprocessing caches here. Large files are ignored by default; keep only lightweight metadata and `.gitkeep` placeholders in version control.

Frame manifests are normally written under `data/splits/`. Paired LR/HR manifests should use matching sequence ids across `--root` and `--hr-root` so the training loader can read real target frames.

Generated degraded LR data should go under `data/processed/`, while reusable degradation settings and metadata can live under `data/degradation/`.
