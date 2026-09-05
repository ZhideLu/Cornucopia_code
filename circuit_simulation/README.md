# Cornucopia logical-memory experiments

Start with `simulate_memory.ipynb` to choose a code, noise probability, memory
basis, cycle count, and decoder settings. The command preview and execution
are separate cells. `retry_bposd.ipynb` applies BP-OSD to stored shots for which
RelayBP has not converged.

| File | Role |
| --- | --- |
| `simulate_memory.py` | Run Cornucopia experiments from the command line |
| `memory_experiment.py` | Circuit preparation, sampling, decoding, and retry calculations |
| `simulation_parameters.py` | Physical conditions and decoder parameters |
| `decoders.py` | RelayBP, BP-OSD, and optional mixed-integer decoding |
| `shot_data.py` | Read and write circuits, decoding matrices, and detector samples |
| `logical_error_statistics.py` | Block failures, observable mismatches, and summary tables |

The BB and surface-code experiments use these same sampling and decoding
routines with their own physical circuits.

From the repository root:

```bash
python circuit_simulation/simulate_memory.py \
  --stage both --codes cornucopia_p21_d6 --basis both --decoding-mode xz \
  --p-list 0.002 --cycles 2 --shots 20 --shot-chunk 10 --workers 2
```

This small run checks the calculation; it is too short to estimate a rare
logical error rate. Results go to `circuit_simulation/results/`: `samples/`
contains circuits and detector/observable samples; `decode/` contains decoder
records and logical-error summaries.

`--stage generate` samples the circuit; `--stage decode` reads saved samples;
`--stage both` runs both steps. Increasing `--shots` extends compatible runs.
Samples are reused only when their circuit, seed, chunk positions, and sizes
match. Noise probabilities must lie in [0, 1]; distinct values must also
have distinct output labels at six significant digits. Cached circuits are
checked against the exact physical parameters before reuse. Changed or
incomplete sample records require regeneration. Resume runs from the same
working directory when stored records use relative paths.

Run the retry notebook after the primary decode has saved unconverged-shot
details. Its physical conditions and primary RelayBP settings must match
the source run. Mathematical definitions, noise channels, and statistical
conventions are in [METHODS.md](../METHODS.md).
