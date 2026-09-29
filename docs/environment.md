# Environment

## CRC Modules

MACE calculations use the CUDA and Intel runtime modules available on the
Notre Dame CRC:

```bash
module purge
module load cuda/12.1 intel/24.2
```

CPU-only postprocessing can use:

```bash
module purge
module load intel/24.2
```

For a VASP NAC calculation on the relaxed primitive cell:

```bash
module purge
module load vasp/6.4.0
```

## Conda Environment

Create the environment from the supplied definition:

```bash
conda env create -f environment.yml
conda activate mlphonon
python -m pip install -e .
```

Set the external data locations before validating the AlGaN configuration:

```bash
export MLPHONON_ROOT=/path/to/MLPhonon
export MLPHONON_PROJECT_ROOT=/path/to/calculation/project
export MACE_MH1_MODEL=/path/to/mace-mh-1.model
export DFT_PHONON_ROOT=/path/to/DFT/Phonon-TC
```

The MACE-MH-1 model checkpoint is an external input. Set its location in the
project YAML file. VASP is required only when Born effective charges and the
dielectric tensor are recalculated for the relaxed structure.

## External Software

- MACE supplies energies, forces, and stresses for relaxation and displaced
  supercells.
- ASE manages structures and calculators.
- phonopy generates harmonic phonons, band structures, and density of states.
- kALDo supplies the finite-displacement IFC cache used by the reference
  calculation.
- phono3py evaluates three-phonon, isotope, alloy-mass, and finite-size
  transport contributions.
- VASP supplies self-consistent Born effective charges and dielectric tensors
  when NAC is enabled in `vasp_relaxed` mode.

Exact versions used for the release are recorded in `environment-lock.yml`.
