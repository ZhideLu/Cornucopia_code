# Physical overhead and routing time

`plot_overhead.py` creates the two-panel overhead/routing-time
figure from the supplied numerical values. Outputs are `fig3.pdf` and
`fig3_data.csv` under `figures/overhead/results/`.

Panel a counts data and measured check qubits:

| Family | Data | Check ancillas | Physical overhead per logical qubit |
| --- | --- | --- | --- |
| Cornucopia | 12P | 6P | 18P/k |
| BB | n | n | 2n/k |
| Rotated surface code | d^2 | d^2-1 | 2d^2-1 |

Panel b uses the supplied routing-cycle totals:

| Distance | 6 | 8 | 10 | 12 | 14 | 16 | 18 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Time (ms) | 10.41 | 11.92 | 12.87 | 13.22 | 14.68 | 15.48 | 16.18 |

`plot_overhead.ipynb` exposes panel dimensions, typography,
markers, axes, labels, and export margins. The default font is DejaVu Sans,
which ships with Matplotlib.

`overhead_data.csv` and `overhead.pdf` are the supplied numerical table and
rendered figure. From the repository root:

```bash
python figures/overhead/plot_overhead.py
```
