# Data

This directory is gitignored: no dataset is committed to the repository.

- **Synthetic data** is not stored — it is regenerated from `inferladder.data` with a fixed
  seed, so every result stays reproducible without shipping files.
- **Real COVID data** is downloaded on demand into this directory, which acts as a local
  cache. Sources and their licenses are documented alongside the loader; nothing is
  redistributed here.
