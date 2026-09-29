# MLPhonon Memory

## Accepted Scope

- DFT endpoint references from auto-kappa calculations.
- MLIP relaxation and finite-displacement FC2/FC3 generation.
- NAC-consistent phonon band structures and density of states.
- phono3py RTA endpoint and aligned binary-alloy transport.
- Natural-isotope and alloy mass-variance scattering.
- Bulk and finite-size calculations.
- Reciprocal-mesh extrapolation using
  `kappa(N) = kappa_inf + S / (N1*N2*N3)^(1/3)`.
- Collection of numerical outputs, literature comparisons, and figures.

## AlGaN Reference Configuration

- Model: MACE-MH-1, `matpes_r2scan` head.
- Endpoint structures: wurtzite AlN and GaN.
- FC2 and FC3 supercells: `5 x 5 x 3`.
- Standard mesh: `16 x 16 x 10`.
- Extrapolation meshes: `12 x 12 x 8`, `14 x 14 x 9`, `16 x 16 x 10`,
  `18 x 18 x 11`, and `20 x 20 x 12`.
- Standard temperature: 300 K.
- Transport backend: phono3py RTA with tetrahedron integration.
- NAC method: Wang.

## Repository Boundary

Numerical results and restart assets are stored outside Git and referenced by
release manifests. JAX-BTE is not part of the MLPhonon source package; a
selected device case may be distributed as an independent archive.
