# Workflow

## Endpoint Calculation

For each endpoint, MLPhonon relaxes the supplied primitive structure with the
selected MLIP, prepares the NAC calculation for the relaxed cell, evaluates
FC2 and FC3 by finite displacement, and constructs phonon band and density-of-
states data. The cached force constants are then mapped to phono3py using
explicit periodic atom matching.

The endpoint transport calculation records the structure, supercell mapping,
NAC tensor, isotope mass variances, integration method, mesh, temperature,
software versions, and input checksums.

## Binary Alloy

The virtual-crystal branch first maps both endpoints to a common primitive
basis and orientation. FC2, FC3, NAC tensors, lattice vectors, internal
coordinates, cation masses, and mass variances are interpolated at each alloy
fraction. Structural and tensor checks are completed before transport.

## Mesh Extrapolation

Three or more symmetry-compatible meshes are evaluated with identical force
constants and scattering settings. Each tensor component is fitted to

```text
kappa(N) = kappa_inf + S / (N1*N2*N3)^(1/3)
```

The collected table reports the sampled conductivities, fitted intercept,
slope, coefficient of determination, and intercept uncertainty.

## Numerical Outputs

The release dataset retains machine-readable conductivity tensors, mode
frequencies, group velocities, lifetimes, mean free paths, spectral and
cumulative conductivity, VCA composition scans, extrapolation parameters, and
the data used for each figure.
