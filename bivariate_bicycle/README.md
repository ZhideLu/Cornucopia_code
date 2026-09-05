# Bivariate-bicycle comparison

`code_construction.py` implements two code instances:

| Name | Parameters | A | B |
| --- | --- | --- | --- |
| `bb_144_12_12` | `[[144,12,12]]`, ell=12, m=6 | x^3+y+y^2 | y^3+x+x^2 |
| `bb_288_12_18` | `[[288,12,18]]`, ell=12, m=12 | x^3+y^2+y^7 | y^3+x+x^2 |

For cyclic shifts $x=S_\ell\otimes I_m$ and $y=I_\ell\otimes S_m$,
$H_X=[A\ B]$ and $H_Z=[B^T\ A^T]$. The circuit module is
`syndrome_circuit.py`. Its schedule is

```text
slot: 1    2   3   4   5   6   7    8
Z:    A1   A3  B1  B2  B3  A2  idle idle
X:    idle A2  B2  B1  B3  A1  A3   idle
```

The simulator shares the Cornucopia sampling and decoding engine, with the
same gate, reset, and measurement noise probabilities.

## Files and calculation

Run memory experiments with `simulate_memory.py` or
`simulate_memory.ipynb`. Use `retry_bposd.ipynb` for stored shots that remain
unresolved after RelayBP decoding.

From the repository root:

```bash
python bivariate_bicycle/simulate_memory.py \
  --stage both --codes bb_144_12_12 --basis Z --decoding-mode xz \
  --p-list 0.002 --cycles 2 --shots 20 --shot-chunk 10 --workers 2
```

Outputs go to `bivariate_bicycle/results/`. See the
[shared simulation notes](../circuit_simulation/README.md) for sampling,
resuming, and retry settings, and [METHODS.md](../METHODS.md) for noise and
logical-error-rate definitions.
