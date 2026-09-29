# AlGaN Literature Data

This directory contains composition-dependent thermal-conductivity curves used
for comparison with the MLPhonon AlGaN calculations. All files use Al fraction
as the first column and thermal conductivity in W m-1 K-1 as the second.

- `dagli_a_axis.csv`: digitized a-axis curve reported by Dagli et al.
- `dagli_c_axis.csv`: digitized c-axis curve reported by Dagli et al.
- `tran_bulk.csv`: digitized bulk AlGaN model curve reported by Tran et al.
- `tran_0p5um.csv`: digitized 0.5 um layer model curve reported by Tran et al.
- `algan_benchmark_300K.csv`: normalized plotting table containing the
  literature curves and their labels.

The numerical coordinates are comparison data and are not included in the
MLIP fitting objective. Dataset labels and citation metadata are recorded in
`metadata.yaml`.
