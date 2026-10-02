# Cornucopia code construction

| File | Role |
| --- | --- |
| `affine_codes.py` | Affine permutations, binary matrices, and algebra over $\mathbb F_2$ |
| `cornucopia_codes.py` | The eight supplied parameter sets and their affine maps |
| `construct_codes.py` | Reconstruct the codes and calculate ranks and check weights |
| `results/matrices/` | $H_X$ and $H_Z$ matrices in Matrix Market format |
| `results/constructed_codes_summary.json`, `.txt` | Algebraic summaries |

Code names encode $P$ and the nominal distance: `cornucopia_p21_d6`
is the $P=21$ instance with parameters $[[252,130,6]]$.

Seven of the eight instances ($P=21,48,75,87,147,192,237$) are the codes
reported in the paper. `cornucopia_p267_d18`, with parameters
$[[3204,1606,18]]$, is an additional distance-18 instance that is not reported
in the paper; the paper's distance-18 code is `cornucopia_p237_d18`.

From the repository root:

```bash
python code_construction/construct_codes.py --write-matrices
```

Results are written to `code_construction/results/`; the copy in the
repository is the output of this command.
The calculation checks
$H_XH_Z^T=0$ and evaluates $k=n-\mathrm{rank}_2(H_X)-\mathrm{rank}_2(H_Z)$.
The `expected_d` field is the supplied distance label; this construction
calculation does not certify it. See [mathematical conventions](../METHODS.md).
