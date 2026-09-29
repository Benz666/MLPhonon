# Data Layout

Calculation outputs are stored in a project directory outside the source
repository:

```text
project/
├── results/
│   ├── AlN/
│   └── GaN/
├── vca/
│   ├── aligned_ifc/
│   ├── qmesh_12x12x8/
│   ├── qmesh_16x16x10/
│   └── extrapolated/
├── figures/
└── manifests/
```

Large restart files are archived separately from compact numerical results.
Every release manifest records the logical dataset name, relative path, size,
SHA256 checksum, producing configuration, and software version.
