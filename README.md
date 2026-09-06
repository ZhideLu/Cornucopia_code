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
| [Figures](figures/) | Logical-error-rate plots, with numerical tables and PDFs |



## Running the calculations

Use Python 3.11 or newer; the examples were checked with Python 3.11.
From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run these commands from the repository root. Each calculation is independent;
the plotting scripts use the included tables and require no prior simulation.

```bash
python code_construction/construct_codes.py --write-matrices
python syndrome_extraction/validate_schedules.py --rounds 2 --write-stim --pretty
python figures/overhead/plot_overhead.py
python figures/logical_error_rates/plot_error_rates.py
```

For a small memory experiment:

```bash
python circuit_simulation/simulate_memory.py \
  --stage both --codes cornucopia_p21_d6 --basis Z --decoding-mode xz \
  --p-list 0.002 --cycles 6 --shots 2000 --shot-chunk 200 --workers 10
```

Results are written to each topic's `results/` directory, which is excluded from Git. 

The supplied matrices, circuits, figure tables, and PDFs are included. Raw Monte Carlo shots and complete decode summaries are not included; regenerating the paper's statistics requires the corresponding simulations.



## License and citation

The code is released under the [MIT License](LICENSE).
