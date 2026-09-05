# Cornucopia code construction

| File | Role |
| --- | --- |
| `affine_codes.py` | Affine permutations, binary matrices, and algebra over $\mathbb F_2$ |
| `cornucopia_codes.py` | The eight supplied parameter sets and their affine maps |
| `construct_codes.py` | Reconstruct the codes and calculate ranks and check weights |
| `matrices/` | Supplied $H_X$ and $H_Z$ matrices in Matrix Market format |
| `constructed_codes_summary.json`, `.txt` | Supplied algebraic summaries |

Code names encode $P$ and the nominal distance: `cornucopia_p21_d6`
is the $P=21$ instance with parameters $[[252,130,6]]$.

From the repository root:

```bash
python code_construction/construct_codes.py --write-matrices
```

Results are written to `code_construction/results/`. The calculation checks
$H_XH_Z^T=0$ and evaluates $k=n-\mathrm{rank}_2(H_X)-\mathrm{rank}_2(H_Z)$.
The `expected_d` field is the supplied distance label; this construction
calculation does not certify it. Both distance-18 instances, $P=237$ and
$P=267$, are retained. See [mathematical conventions](../METHODS.md).
