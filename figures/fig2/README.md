# Fig. 2: logical error rates

This directory produces one two-panel figure, `fig2.pdf` (Fig. 2 of the paper).
Panel a shows the seven Cornucopia codes. Panel b compares Cornucopia and BB
codes at distances 12 and 18 with surface codes at distances 13 and 19.
Every family uses X/Z-averaged results.

| File | Role |
| --- | --- |
| `plot_fig2.py`, `plot_fig2.ipynb` | Reproduce the figure from the supplied CSV tables |
| `refit_fig2.ipynb` | Read three decode summaries, export seven averaged/fit tables, and draw the same figure |
| `data/` | The seven supplied tables listed below |
| `fig2.pdf` | The supplied figure |

From the repository root:

```bash
python figures/fig2/plot_fig2.py
```

`fig2.pdf` is written into this folder, replacing the supplied copy. Basic plotting
needs Matplotlib and NumPy, but no raw shot files, decode summaries, or decoder
installation. The plotting notebook exposes the final figure's display
settings. Both plotting entry points use the same renderer and DejaVu Sans.

The seven tables are:

- `fig2_cornucopia_xz_average_rates.csv`
- `fig2_cornucopia_xz_average_fit_data.csv`
- `fig2_cornucopia_xz_average_fit_coefficients.csv`
- `fig2_bb_xz_average_rates.csv`
- `fig2_bb_xz_average_fit_data.csv`
- `fig2_bb_xz_average_fit_coefficients.csv`
- `fig2_surface_xz_average_rates.csv`

The refitting notebook requires `results/decode/summary.txt` under
`circuit_simulation/`, `bivariate_bicycle/`, and `surface_code/`; these are
written by the simulation runners and are not included. It replaces the seven
tables in `data/` and `fig2.pdf` with the refitted versions;
`git checkout figures/fig2` restores the supplied files.

Fits use unweighted least squares in log space with the model

$$\log p_L=\frac d2\log p+c_0+c_1p+c_2p^2.$$

Cornucopia fits use $p\leq0.0025$; BB and surface fits use $p\leq0.0045$.
Panel a also displays Cornucopia points beyond its fit range, while panel b
shows only Cornucopia points within that range. The surface curves are refit
from their averaged rate table by the shared renderer.

The reported effective rate is $p_L=1-(1-P_B)^{1/(kT)}$, with X/Z averaging
performed on block failure probabilities before conversion. Error bars use
the counting approximation $p_L/\sqrt{N_{\mathrm{fail},X}+N_{\mathrm{fail},Z}}$.
See [mathematical conventions](../../METHODS.md) for its interpretation and
the limits of extrapolation.
