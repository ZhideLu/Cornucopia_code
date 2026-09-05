# Mathematical and numerical conventions

The affine maps `F_i` and `G_i` in `code_construction/cornucopia_codes.py`
correspond to the connectivity operators `A_i` and `B_i` in the
[paper](https://arxiv.org/abs/2608.02773).



## Circuits and noise

The Cornucopia schedule measures X and Z checks in parallel over 12 CX layers.
Each layer is validated against the parity-check matrices and checked for
qubit collisions. Logical memory is prepared and measured in either the X or
Z basis. The BB implementation uses eight schedule slots; the rotated
surface-code circuit has four CX layers per round.

The memory simulation sets

$$p_{\rm CX}=p_{\rm measurement}=p_{\rm final\ measurement}=p_{\rm reset}=p.$$

A two-qubit depolarizing channel follows each CX. Measurement and reset
errors are represented by Pauli flips in the complementary basis. No idle
noise or single-qubit Clifford depolarization is added by these runners.
`xyz` retains all detector types; `xz` retains the CSS detector type relevant
to the chosen memory basis. This switch changes the decoding problem, not
the physical circuit noise.

The circuit-distance search has a different purpose and noise setting:
it uses CX faults to construct detector-error-model columns, with measurement
and reset flip probabilities set to zero. A random logical constraint selects
a nontrivial undetectable fault pattern. Its weight bounds circuit distance
from above. Finite randomized search cannot establish optimality.

## Logical-error reporting

A shot is one complete memory experiment, from preparation through $T$
syndrome cycles to final readout. It fails if at least one decoded logical
observable is incorrect. Let
$P_B$ denote this block failure probability, $k$ the observable count, and $T$
the cycle count. The effective rate used here is

$$p_L=1-(1-P_B)^{1/(kT)}.$$

In the CSV tables, `LER` denotes $P_B$, `observables` denotes $k$, and
`LER_per_cycle_per_logical` denotes $p_L$.

This conversion assumes a factorized effective rate; it does not demonstrate
independence of logical errors. The direct observable-mismatch rate is a
separate statistic in the exported tables. X/Z-averaged results first average
the matched X- and Z-basis block rates, then apply the conversion.

The plotted error bars use $p_L/\sqrt{N_{\rm fail}}$, with summed
failure counts for X/Z averages. This is a counting approximation;
it is not an exact binomial confidence interval. Zero-failure points are not
replaced with artificial positive estimates for logarithmic plotting.

The fit model is

$$\log p_L=\frac d2\log p+c_0+c_1p+c_2p^2.$$

Each code has its own coefficients, obtained by unweighted least squares
in log space over the ranges specified in the notebooks. Curves beyond the
sampled region are extrapolations. The plotted data and fits alone do not constitute
an asymptotic threshold estimate.
