# Fig. 3: physical overhead and routing time

`plot_fig3.py` creates panels a and b of Fig. 3 (physical overhead and routing time) from the supplied numerical values. It writes `fig3.pdf` and `fig3_data.csv` into this folder, replacing the supplied copies.

Panel a counts data and measured check qubits:

| Family | Data | Check ancillas | Physical overhead per logical qubit |
| --- | --- | --- | --- |
| Cornucopia | 12P | 6P | 18P/k |
| BB | n | n | 2n/k |
| Rotated surface code | d^2 | d^2-1 | 2d^2-1 |

Panel b uses the supplied routing-cycle totals:

| Distance | 6 | 8 | 10 | 12 | 14 | 16 | 18 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Time (ms) | 10.33 | 11.84 | 12.79 | 13.14 | 14.60 | 15.41 | 16.10 |



From the repository root:

```bash
python figures/fig3/plot_fig3.py
```
