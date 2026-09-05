# Rotated surface-code comparison

`code_construction.py` accepts names `surface_dN` with integer $N\ge2$,
representing $[[d^2,1,d]]$ rotated planar codes. The `all` shortcut selects
$d=6,8,10,12,14,16,18$.

`syndrome_circuit.py` starts from Stim's `surface_code:rotated_memory_x`
or `surface_code:rotated_memory_z` generator. Four parallel CX layers form
each syndrome round. Ancilla Hadamard gates are absorbed into equivalent
basis-specific reset and measurement operations:

```text
R - H       -> RX
H - MR - H  -> MRX
```

Measurement-record order is preserved. The same CX depolarization and
measurement/reset flips used for Cornucopia are then applied. No additional
single-qubit Clifford depolarization is inserted.

In `xz` mode the relevant detector subset is selected by lattice coordinates:
Z checks have $(x+y)\bmod4=0$, and X checks have $(x+y)\bmod4=2$.

## Files and calculation

Run memory experiments with `simulate_memory.py` or
`simulate_memory.ipynb`. Use `retry_bposd.ipynb` for stored shots that remain
unresolved after RelayBP decoding.

From the repository root:

```bash
python surface_code/simulate_memory.py \
  --stage both --codes surface_d3 --basis Z --decoding-mode xz \
  --p-list 0.002 --cycles 2 --shots 20 --shot-chunk 10 --workers 2
```

Outputs go to `surface_code/results/`. See the
[shared simulation notes](../circuit_simulation/README.md) for sampling,
resuming, and retry settings, and [METHODS.md](../METHODS.md) for noise and
logical-error-rate definitions.
