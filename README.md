# MLPhonon

MLPhonon is a configuration-driven workflow for calculating phonon properties
and lattice thermal transport with machine-learning interatomic potentials.
It connects DFT endpoint references, MLIP structural relaxation, non-analytical
corrections, finite-displacement force constants, phonon band structures,
density of states, phono3py transport, virtual-crystal alloys, and reciprocal-
mesh extrapolation.

The first reference application is wurtzite AlN, GaN, and AlGaN using the
MACE-MH-1 MATPES-r2SCAN head. Numerical datasets and figures are distributed
separately from the source code.

## Workflow

```text
structure and model configuration
-> MLIP relaxation
-> Born charges and dielectric tensor
-> FC2 and FC3
-> phonon bands and density of states
-> endpoint RTA transport
-> aligned virtual-crystal alloy transport
-> q-mesh extrapolation
-> numerical collection and figures
```

The transport implementation uses phono3py with natural-isotope scattering,
optional alloy mass variance, tetrahedron or Gaussian integration, and
mode-resolved output at user-selected temperatures. The current IFC cache is
generated through the kALDo finite-displacement representation and converted
to explicitly atom-matched phono3py tensors before transport.

## Installation

The CRC software modules and conda setup are documented in
[`docs/environment.md`](docs/environment.md). After activating the environment,
install the repository in editable mode:

```bash
python -m pip install -e .
```

## Configuration

Start from [`configs/algan_mh1_matpes_r2scan.yaml`](configs/algan_mh1_matpes_r2scan.yaml).
Every model path, endpoint structure, DFT reference root, supercell, q mesh,
temperature, and output location is declared in the YAML file.

Validate a configuration before running a calculation:

```bash
mlphonon-validate configs/algan_mh1_matpes_r2scan.yaml
```

On the Notre Dame CRC, the reference paths can be selected with:

```bash
export MLPHONON_ROOT=/groups/tluo/bma/WorkFlows/MLPhonon
export MLPHONON_PROJECT_ROOT=/groups/tluo/bma/LM/AlGaN/MACE_LD/AlGaN_MACE_kaldo
export MACE_MH1_MODEL=/groups/tluo/bma/LM/AlGaN/MACE_LD/mace-mh-1.model
export DFT_PHONON_ROOT=/groups/tluo/bma/DFTDatasets/Phonon-TC
```

The individual scientific stages are exposed as Python entry points under
`scripts/`. This keeps execution transparent on workstations and computing
clusters without embedding a scheduler in the package. Complete command
sequences are given in [`docs/usage.md`](docs/usage.md).

The source and numerical release layout is described in
[`docs/release.md`](docs/release.md).

## Data Policy

The Git repository contains source code, configuration, documentation, and
small literature datasets. Model checkpoints, VASP POTCAR files, force
constants, solver arrays, and full numerical results are external assets. Their
locations and SHA256 checksums are recorded in release manifests.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff).
