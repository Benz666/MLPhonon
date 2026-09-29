# Usage

## Configuration

Export the four external locations described in `docs/environment.md`, then
validate the project file:

```bash
mlphonon-validate configs/algan_mh1_matpes_r2scan.yaml
```

The model label used by the reference configuration is
`mh1_matpes_r2scan`. Endpoint material names are `AlN` and `GaN`.

## Endpoint Stages

Run the scientific stages in order for each endpoint:

```bash
python scripts/01_relax.py CONFIG --model mh1_matpes_r2scan --material AlN
python scripts/02_prepare_nac.py CONFIG --model mh1_matpes_r2scan --material AlN
python scripts/03_compute_ifc.py CONFIG --model mh1_matpes_r2scan --material AlN
python scripts/04_banddos.py CONFIG --model mh1_matpes_r2scan --material AlN
python scripts/05_endpoint_rta.py CONFIG --model mh1_matpes_r2scan --material AlN
```

Repeat the sequence with `--material GaN`. The NAC stage prepares a VASP
LEPSILON calculation on the MLIP-relaxed structure and converts its
`vasprun.xml` to `BORNINFO.mlip_relaxed` when `nac.run_vasp` is enabled.

## Aligned AlGaN VCA

Construct orientation-aligned virtual-crystal IFCs from the two endpoint
directories:

```bash
python scripts/12_build_aligned_algan_vca.py \
  --endpoint-aln ALN_ENDPOINT \
  --endpoint-gan GAN_ENDPOINT \
  --output-root VCA_IFC_ROOT \
  --fraction 0.125 --fraction 0.23 --fraction 0.25 \
  --fraction 0.375 --fraction 0.5 --fraction 0.625 \
  --fraction 0.75 --fraction 0.875 \
  --fc2-supercell 5 5 3 --fc3-supercell 5 5 3
```

Evaluate each composition with `scripts/11_phono3py_vca.py`. Its inputs are
the aligned VCA directory, the endpoint IFC directories, and the corresponding
phono3py endpoint results. The standard mesh is `16 16 10` at 300 K.

Collect the composition scan with:

```bash
python scripts/14_collect_phono3py_vca.py \
  --vca-root VCA_RESULT_ROOT \
  --gan-result GAN_RESULT_JSON \
  --aln-result ALN_RESULT_JSON \
  --benchmark-csv references/literature/algan_benchmark_300K.csv \
  --dft-gan-kappa 210.582 --dft-aln-kappa 256.656
```

## Reciprocal-Mesh Extrapolation

Use `scripts/15_phono3py_endpoint_qmesh.py` to evaluate endpoint meshes. Run
the aligned VCA calculation on the same mesh ladder. Collect all endpoint and
VCA results with `scripts/16_collect_phono3py_qmesh_extrapolation.py`.

The collector requires at least three meshes and fits every conductivity
component to:

```text
kappa(N) = kappa_inf + S / (N1*N2*N3)^(1/3)
```

The extrapolated composition dependence is represented with the reduced
fixed-temperature Dagli form:

```text
kappa(x) = 1 / [(1-x)/kappa_GaN + x/kappa_AlN + A*x^p*(1-x)^q]
```

## Cleanup

After a transport case has written its compact tables and result summary,
remove solver scratch arrays with:

```bash
python scripts/09_cleanup.py CONFIG --model mh1_matpes_r2scan --material MATERIAL
```

Force constants and compact phono3py results are retained as restart assets.
