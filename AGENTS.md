# Repository Guide

Read `README.md`, `docs/workflow.md`, `docs/environment.md`, and `MEMORY.md`
before changing the workflow.

## Scope

- Keep this repository focused on DFT references, MLIP lattice dynamics,
  phono3py transport, virtual-crystal alloys, and numerical postprocessing.
- Device-level JAX-BTE development is maintained separately.
- Do not commit model checkpoints, VASP POTCAR files, force constants, solver
  arrays, or project result directories.
- Preserve atom ordering, primitive-cell orientation, NAC provenance, and
  reciprocal-mesh definitions in every data conversion.

## Project Layout

- `mlphonon/`: reusable Python implementation.
- `scripts/`: explicit calculation and collection entry points.
- `configs/`: reproducible project configurations.
- `references/literature/`: curated comparison data and citations.
- `docs/`: methods, environment, data layout, and usage.
- `manifests/`: checksums and external data inventories for releases.

## Changes

Keep scientific defaults in configuration files rather than hard-coding them.
Update `MEMORY.md` when accepted methods, file contracts, or release assets
change. Documentation should describe the supported workflow and current
results without development chronology.
