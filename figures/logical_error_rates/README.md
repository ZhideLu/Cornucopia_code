# Logical error rates

This directory produces one two-panel figure, `stacked_error_rate_fits.pdf`.
Panel a shows the seven Cornucopia codes. Panel b compares Cornucopia and BB
codes at distances 12 and 18 with surface codes at distances 13 and 19.
Every family uses X/Z-averaged results.

| File | Role |
| --- | --- |
| `plot_error_rates.py`, `.ipynb` | Reproduce the stacked figure from the supplied CSV tables |
| `refit_error_rates.ipynb` | Read three decode summaries, export seven averaged/fit tables, and draw the same figure |
| `data/` | The seven supplied tables listed below |
| `pdfs/stacked_error_rate_fits.pdf` | The supplied stacked figure |

From the repository root:

```bash
python figures/logical_error_rates/plot_error_rates.py
```

The PDF is written to `figures/logical_error_rates/results/`. Basic plotting
needs Matplotlib and NumPy, but no raw shot files, decode summaries, or decoder
installation. The plotting notebook exposes the final figure's display
settings. Both plotting entry points use the same renderer and DejaVu Sans.

The seven tables are:

- `cornucopia_xz_average_rates.csv`
- `cornucopia_xz_average_fit_data.csv`
- `cornucopia_xz_average_fit_coefficients.csv`
- `bb_xz_average_rates.csv`
- `bb_xz_average_fit_data.csv`
- `bb_xz_average_fit_coefficients.csv`
- `surface_xz_average_rates.csv`

The refitting notebook requires `results/decode/summary.txt` under
`circuit_simulation/`, `bivariate_bicycle/`, and `surface_code/`. These raw
summaries are not included. It writes seven updated CSVs and the single stacked
PDF to `results/`, leaving the supplied tables in `data/` untouched.

Fits use unweighted least squares in log space with the model

$$\log p_L=\frac d2\log p+c_0+c_1p+c_2p^2.$$

Cornucopia fits use $p\leq0.0025$; BB and surface fits use $p\leq0.005$.
Panel a also displays Cornucopia points beyond its fit range, while panel b
shows only Cornucopia points within that range. The surface curves are refit
from their averaged rate table by the shared renderer.

The reported effective rate is $p_L=1-(1-P_B)^{1/(kT)}$, with X/Z averaging
performed on block failure probabilities before conversion. Error bars use
the counting approximation $p_L/\sqrt{N_{\mathrm{fail},X}+N_{\mathrm{fail},Z}}$.
See [mathematical conventions](../../METHODS.md) for its interpretation and
the limits of extrapolation.
