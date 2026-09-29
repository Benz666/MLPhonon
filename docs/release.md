# Release Packaging

MLPhonon source releases are created from a clean Git commit. Numerical
results, DFT references, and downstream device examples are distributed as
separate archives so that source history remains small and reviewable.

The standard publication bundle contains:

1. The MLPhonon Git source archive.
2. Compact phono3py endpoint, VCA, and q-mesh extrapolation results.
3. Distilled auto-kappa endpoint references and the matching open-source
   auto-kappa version.
4. A separately versioned downstream device-transport case when required.

Each archive has a SHA256 checksum. Numerical archives also contain a
file-level checksum list. Licensed pseudopotentials, model checkpoints, raw
electronic-structure scratch data, scheduler files, and transient solver
caches are excluded.

For NAS storage, transfer the completed top-level ZIP and its checksum with
`rsync --partial --progress`; verify the checksum at the destination before
removing any local copy.
