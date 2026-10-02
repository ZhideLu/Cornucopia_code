# Syndrome extraction

| File | Role |
| --- | --- |
| `syndrome_schedule.py` | Twelve parallel CX layers for Cornucopia checks |
| `memory_circuit.py` | Logical preparation, repeated syndrome extraction, and final readout |
| `validate_schedules.py` | Check layer reconstruction, qubit collisions, and noiseless detectors |
| `results/stim_circuits/` | Eight noiseless, two-round Stim circuits |
| `results/schedule_validation_summary.json` | Schedule checks; circuit paths are relative to this file |

From the repository root:

```bash
python syndrome_extraction/validate_schedules.py --rounds 2 --write-stim --pretty
```

Circuits and summaries are written to `syndrome_extraction/results/`; the
copy in the repository is the output of this command.
The BB and surface-code circuits are in their respective code-family folders.
See [mathematical conventions](../METHODS.md) for layer and noise conventions.
