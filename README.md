# Cornucopia

Code accompanying **Quantum error correction at ultra-low overhead**,
Zhide Lu, Weikang Li, and Dong-Ling Deng (2026).
[arXiv:2608.02773](https://arxiv.org/abs/2608.02773).

Code construction and circuit-level simulations of Cornucopia codes, with bivariate-bicycle and rotated surface codes for comparison.

| Topic | Contents |
| --- | --- |
| [Code construction](code_construction/) | Affine maps, code parameters, and parity-check matrices |
| [Syndrome extraction](syndrome_extraction/) | Parallel CX schedules |
| [Circuit distance](circuit_distance/) | Randomized BP-OSD search for circuit-distance upper bounds |
| [Circuit simulation](circuit_simulation/) | Cornucopia memory experiments |
| [Bivariate-bicycle codes](bivariate_bicycle/) | BB construction, syndrome circuit, and memory experiments |
| [Surface codes](surface_code/) | Rotated surface-code construction, syndrome circuit, and memory experiments |
| [Figures](figures/) | Fig. 2 (logical error rates) and Fig. 3 (overhead and routing time), with numerical tables and PDFs |



## Running the calculations

Use Python 3.11 or newer; the examples were checked with Python 3.11.
From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Tests

The test suite checks the code construction, syndrome circuits, sampling and logical-error statistics, the figure tables and plots, and the command-line workflows on small inputs. From the repository root, run

```bash
python -m pytest
```

Use `python -m pytest` rather than the bare`pytest` command: the tests import the topic folders (`code_construction`,`circuit_simulation`, ...) as packages, and `python -m` puts the repository root on the import path. To run a single file, pass its path, for example`python -m pytest tests/test_codes.py`.

### Calculations

Run these commands from the repository root. Each calculation is independent; the plotting scripts use the included tables and require no prior simulation.

```bash
python code_construction/construct_codes.py --write-matrices
python syndrome_extraction/validate_schedules.py --rounds 2 --write-stim --pretty
python figures/fig2/plot_fig2.py
python figures/fig3/plot_fig3.py
```

For a small memory experiment:

```bash
python circuit_simulation/simulate_memory.py \
  --stage both --codes cornucopia_p21_d6 --basis Z --decoding-mode xz \
  --p-list 0.002 --cycles 6 --shots 2000 --shot-chunk 200 --workers 10 \
  --gamma0 0.1 --pre-iter 200 --num-sets 20 --set-max-iter 100 --stop-nconv 1 \
  --relaybp-fallback --relaybp-fallback-gamma0 0.1 --relaybp-fallback-pre-iter 500 \
  --relaybp-fallback-num-sets 200 --relaybp-fallback-set-max-iter 200 \
  --relaybp-fallback-stop-nconv 1
```

The calculation and simulation scripts write to a `results/` folder inside their topic folder. The `results/` folders of `code_construction/` and`syndrome_extraction/` are part of the repository: they contain the output of the first two commands above, that is, the parity-check matrices, the Stim circuits, and their summaries. The `results/` folders of `circuit_distance/` and of the memory experiments (`circuit_simulation/`, `bivariate_bicycle/`, and `surface_code/`) are created on the first run and ignored by Git (`.gitignore`). The figure scripts have no`results/` folder; they rewrite `fig2.pdf`, `fig3.pdf`, and `fig3_data.csv` in`figures/fig2/` and `figures/fig3/`.

The supplied matrices, circuits, figure tables, and PDFs are included. Raw Monte Carlo shots and complete decode summaries are not included; regenerating the paper's statistics requires the corresponding simulations.

## License and citation

The code is released under the [MIT License](LICENSE). If you use it, please
cite the paper:

```bibtex
@misc{lu2026cornucopia,
  title         = {Quantum error correction at ultra-low overhead},
  author        = {Lu, Zhide and Li, Weikang and Deng, Dong-Ling},
  year          = {2026},
  eprint        = {2608.02773},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2608.02773}
}
```
