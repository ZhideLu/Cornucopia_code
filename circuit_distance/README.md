# Circuit-distance upper bounds

`estimate_distance.py` searches for an undetectable fault with a nontrivial
logical effect. Each trial adds a random logical constraint to the detector
problem and solves it with BP-OSD. The minimum fault weight found is an upper
bound on circuit distance. A finite randomized search cannot certify optimality.

`estimate_distance.ipynb` exposes the code, memory basis, trial count, and
decoder parameters, then shows and executes the corresponding command.

From the repository root, a small search is:

```bash
python circuit_distance/estimate_distance.py \
  --codes cornucopia_p21_d6 --basis Z --num-trials 20 \
  --workers 2 --chunk-size 10 --osd-order 1 --max-iter 20
```

Outputs go to `circuit_distance/results/`, which is created on the first run
and ignored by Git. This small search only demonstrates the workflow; for
`cornucopia_p21_d6` it typically finds the upper bound 6, above the bound 5 in
Table 1 of the paper. The bounds in Table 1 come from much longer searches,
configured as in `estimate_distance.ipynb`. Running the search again appends trials to
`*_trials.jsonl` and updates `summary.json` and `summary.txt`. This search uses CX faults;
measurement and reset flips are set to zero when constructing its detector
error model. The memory-simulation noise model is described separately in
[mathematical conventions](../METHODS.md).
