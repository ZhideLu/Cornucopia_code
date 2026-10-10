# Mathematical and numerical conventions

The affine maps `F_i` and `G_i` in `code_construction/cornucopia_codes.py`
correspond to the connectivity operators `A_i` and `B_i` in the
[paper](https://arxiv.org/abs/2608.02773).

## Coordinates and permutation matrices

Each block contains $P=3q$ qubits, and every supplied instance has
$\gcd(q,3)=1$. The code labels the qubits of a block by $z\in\mathbb Z_P$,
whereas the paper uses grid coordinates $(x,y)\in\mathbb Z_3\times\mathbb Z_q$
(row $x$, column $y$). The two labels are related by the Chinese remainder
theorem,

$$(x,y)=(z \bmod 3,\ z \bmod q).$$

An affine map `(a, b)`, $z\mapsto az+b \pmod P$, therefore acts as
$(x,y)\mapsto(ax+b \bmod 3,\ ay+b \bmod q)$. In all supplied instances
$a\equiv 1 \pmod q$, so the column part is the shift $y\mapsto y+(b \bmod q)$,
and the row part is $x\mapsto x+1$ for `F_0` and `G_2`, $x\mapsto 2x+2$ for
`F_1` and `G_3`, and $x\mapsto x$ otherwise (Eqs. 3-5 of the paper). For
example, `(80, 128)` in `cornucopia_p237_d18` ($q=79$) is
$(x,y)\mapsto(2x+2,\ y+49)$, the map $A_1$ of Extended Data Table 1.

`permutation_matrix(a, b, P)` in `code_construction/affine_codes.py` returns
the matrix $M$ with $M_{z,\,az+b}=1$: row $z$ has its single 1 in column
$f(z)$. This is the paper's convention
$(\mathcal A_k)_{u,u'}=1 \iff u'=A_k(u)$ (Methods, after Eq. 2). With
$H_X=[F_{j-i}\mid G_{j-i}]$ and $H_Z=[G_{i-j}^{T}\mid F_{i-j}^{T}]$, the X check
at local index $z$ in check block $i$ acts on $F_{j-i}(z)$ in data block $L_j$
and on $G_{j-i}(z)$ in $R_j$, and the Z check at $z$ acts on $G_{i-j}^{-1}(z)$
in $L_j$ and on $F_{i-j}^{-1}(z)$ in $R_j$. The CX layers built in
`syndrome_extraction/syndrome_schedule.py` use the same pairing ($f(z)$ for X
layers, $f^{-1}(z)$ for Z layers).

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

A shot is one complete memory experiment, from preparation through $T$ syndrome cycles to final readout. It fails if at least one decoded logical observable is incorrect. Let $P_B$ denote this block failure probability, $k$ the observable count, and $T$
the cycle count. The effective rate used here is

$$p_L=1-(1-P_B)^{1/(kT)}.$$

In the `*_xz_average_rates.csv` tables, `LER` denotes the X/Z-averaged $P_B$,
`x_LER` and `z_LER` the single-basis block failure probabilities, and
`LER_per_cycle_per_logical` and `yerr` denote $p_L$ and its error bar. In the
`*_fit_data.csv` tables, `y` and `yerr` are $p_L$ and its error bar.

This conversion assumes a factorized effective rate; it does not demonstrate
independence of logical errors. The decode summaries written by the simulation
runners (`summary.txt` and `summary.json`) also report the direct
observable-mismatch rate `direct_per_logical_LER` as a separate statistic; it
is not part of the supplied CSV tables. X/Z-averaged results first average
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
