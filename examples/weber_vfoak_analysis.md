# VFOAK: the ballot rule works; the quoted effectiveness column does not

## Conclusion and scope

For independent Uniform(0,1) voter/candidate utilities, Weber's stated
large-electorate optimal-strategy criterion, and the two allowed ballots
**+1 on the top k OR −1 on the bottom k**, the exact effectiveness values are:

| Candidates m | Best k | Exact effectiveness (%) | Quoted column (%) |
| ---: | ---: | ---: | ---: |
| 2 | 1 | 81.6497 | 81.65 |
| 3 | 1 | 87.5000 | 87.50 |
| 4 | 1 | 80.8290 | 80.83 |
| 5 | 2 | **85.2058** | **86.96** |
| 6 | 2 | **84.1187** | **86.25** |
| 10 | 4 | **85.3099** | **88.09** |

The disagreement cannot be fixed by finding a better symmetric utility-based
strategy with those allowed ballots: the rule below already maximizes the
relevant objective pointwise, and all allowed ballots have the same centered
squared length. The last three quoted values exceed the resulting bound.

**Source verification, updated after reviewing the supplied scans:** The
user supplied images of printed pages 14, 15, 16, and 19 of *Reproducing
Voting Systems*, in [Weber's collection, CFDP 498,
1978](https://elischolar.library.yale.edu/cgi/viewcontent.cgi?article=1731&context=cowles-discussion-paper-series).
These directly confirm, without relying on OCR:

- Page 14: H is defined by descending utilities and the middle-utility
  average being <= the overall mean; the text asserts “This region is a
  simplex” and prints the erroneous centroid analyzed below.
- Page 15: the effectiveness formula is exactly the one encoded in the
  repository, and it is explicitly obtained from the page-14 centroids.
- Page 16: the claimed optimal fraction .368 and limit 92.25% are obtained
  from that same erroneous formula.
- Page 19: the published table does contain 86.96%, 86.25%, and 88.09% for
  m=5, 6, and 10, respectively, and 92.25% in the limit.

The previous source-access/OCR caveat is therefore resolved for these
claims. The error is present in the printed paper, not introduced by OCR
or the repository's transcription.

## The specific error on page 71: H is not a simplex

Take m=5, k=2 and descending utilities. The closure of H is

    1 >= u1 >= u2 >= u3 >= u4 >= u5 >= 0,
    u3 <= (u1+u2+u3+u4+u5)/5.

Taking the closure does not change its volume or centroid. The original
ordered cube is a 5-simplex with six vertices v_j=(1 repeated j times,
0 repeated 5-j times). The inequality cuts this simplex. Its defining
linear functional, proportional to A-B, is

    F(u) = u1+u2-4*u3+u4+u5.

F(v_j), j=0,...,5, is 0, 1, 2, -2, -1, 0. There are **two positive and
two negative original vertices**. Every edge connecting a positive vertex
to a negative vertex contributes a new vertex on the cutting hyperplane.
Consequently the cut region has **eight vertices**, not six:

    (0, 0,   0,   0,   0)
    (1, 0,   0,   0,   0)
    (1, 1,   0,   0,   0)
    (1, 1,   1/2, 0,   0)
    (1, 1/2, 1/2, 1/2, 0)
    (1, 1,   1,   1,   1)
    (1, 1/3, 1/3, 0,   0)    <-- additional vertex
    (1, 1,   2/3, 2/3, 0)    <-- additional vertex

It is full-dimensional in R^5 and has eight extreme points, whereas a
5-simplex has exactly six. Thus it cannot be a simplex. In particular,
the last two vertices cannot be dropped when calculating its centroid.

The first six points are the one-vertex-per-zero-count list printed on
page 14, specialized to m=5. Treating those six points as a simplex gives
their arithmetic mean, which also equals the printed formula for h at m=5:

    h_wrong = (5/6, 7/12, 1/3, 1/4, 1/6).

Averaging the first two coordinates for a1 and the remaining three for b1
then gives

    a1_wrong = 17/24, b1_wrong = 1/4,
    omega_wrong = a1_wrong-b1_wrong = 11/24,
    effectiveness_wrong = (11/24)*sqrt(18/5) = 86.96263565...%.

**This reproduces the disputed number exactly.** It is not just a generic
objection to the geometry.

The true centroid, obtained independently by exact rational integration
of a triangulation of the eight-vertex polytope, is

    h_true = (5/6, 67/108, 41/108, 31/108, 1/6).
    a1_true = 157/216, b1_true = 5/18,
    omega_true = 97/216,
    effectiveness_true = (97/216)*sqrt(18/5) = 85.20581473...%.

Its volume is 1/240, exactly half the ordered cube's volume 1/120, as
symmetry requires. The rational triangulation calculation is tested
independently of the order-statistic integration used elsewhere here.

The simplex assertion does hold for the nondegenerate k=1 regions at m=3
and m=4: only one original vertex is strictly on each side of the cutting
hyperplane, so only one new vertex is introduced. This explains why the
error is invisible in the first small cases.

### Separate apparent typographical error in the printed table

For odd m, the page-14 entry for b2 prints

    b2 = [2 + (3/2)(k−1)] / [8(m+1)].

Its denominator should be 2(m+1) to agree with b2=1−a1, the symmetry
u -> 1−u, and the equality a1−b1=a2−b2 asserted on page 15. The even-m
entry already has 2(m+1). For example, at m=5,k=2 the printed formulas
give a1=17/24, b1=1/4, a2=3/4, b2=7/96, so their two gaps are not equal.
Changing the odd-m b2 denominator to 2(m+1) gives b2=7/24 and restores
equality of the two *printed* gaps, both 11/24.

This is separate from the substantive geometry error. Correcting this
typo alone leaves the incorrect 86.96% effectiveness unchanged: page 15's
formula already uses the gap a1−b1. The true gap is 97/216, not 11/24.

## Correct ballot rule

Let utilities be sorted **descending**, u1 >= ... >= um, and let ubar be their
arithmetic mean. Fix k for the voting system, not individually for each voter.
It suffices to consider 1 <= k <= floor(m/2): replacing k by m-k gives an
equivalent system under additive ballot normalization.

Compute

- A = sum of (ui − ubar) for the top k candidates;
- B = −sum of (ui − ubar) for the bottom k candidates.

If A >= B, vote +1 for the top k. Otherwise vote −1 against the bottom k.
Utility ties and A=B can be broken symmetrically; they have probability zero
in the continuous model, except for the equivalent choices when m=2k.

Equivalently, use ordinary approval ballots with **k OR m-k approvals**.
Adding 1 to EVERY entry of an against ballot changes no candidate score
differences, even when the added constant differs from voter to voter.

For m>2k, the equivalent central-utilities rule is:

- central average < ubar: approve the top k;
- central average > ubar: approve the top m-k.

Indeed B−A is the sum of the centered middle m−2k utilities. When m=2k,
both choices are the same approval ballot. The current branch's
`vote_for_or_against_k` implements this rule correctly for its documented
positive-k domain. The new tests independently enumerate every eligible
ballot for small m and verify optimality.

For five candidates and k=2, the choices are 11000 and 00111 with negative
signs on the latter's ones, or equivalently 11000 and 11100 after ranking.
This is restricted-cardinality approval, **not unrestricted approval** and
not simultaneous +top-k/−bottom-k.

## Which repository implementation is correct?

I inspected the actual source at both commits, and the current PR #57 diff:

- **6a201e5b:** `elsim/strategies/strategies.py` implements the correct
  central-average choice, with approval-normalized ballots, for integer
  1 <= k <= floor(m/2). These are the ballots checked by the exhaustive
  optimality tests. It is not merely the implementation closest to the
  printed numbers: it implements the stated optimal strategy. The
  surrounding historical effectiveness formulas still require correction.
- **7af1834e:** the active code assigns the first floor(n/2) voters to vote
  for, and all remaining voters to vote against, independently of their
  utilities. Earlier mean/midrange experiments remain in the file but do
  not control the choice. This is not the optimal strategy. The commit
  itself changes a test; this description concerns the implementation in
  its source tree.
- **PR #57:** independently flips a fair coin for each voter, choosing
  +top-k or -bottom-k. This is also not the optimal strategy. It has the
  same utility-independent choice problem as 7af1834e, although a fixed
  half/half split and independent coins are not literally the same finite
  election model. Under independent uniform utilities they have the same
  limiting effectiveness, equal to the vote-for-k effectiveness: both
  signs have the same centered covariance and ballot length, without the
  extra gain from choosing the better sign for the particular utilities.

Signed ballots are not the problem. A correct signed implementation and
6a201e5b's normalized approval implementation are election-equivalent.
The problem in the older implementation and PR #57 is **how each voter
chooses which ballot type to use**.

The statement “a voter is equally likely to be of either type” describes
the distribution induced by optimal choices under uniform utilities. It
does not authorize choosing the type independently of those utilities.

## Why an ordinary election simulation IS the right model

In the section “Optimal voting strategies,” Weber explicitly defines an
optimal ballot as maximizing sum_c w_c (u_c − ubar). In “The effectiveness
of a voting system,” he defines effectiveness as

    lim[n→∞] (E[utility of elected candidate] − n/2)
             / (E[maximum candidate utility] − n/2).

Those are exactly a ballot strategy and an asymptotic election-outcome SUE.
They do not require drawing random ballot types independently of utilities,
nor a different welfare metric for reproducing systems.

Replacing each voter's utilities with the conditional mean utilities for
that voter's ballot type discards residual utility variation. A resulting
100% SUE in that different model does NOT show that Weber's definition is
something other than election-outcome SUE: it changes the utility-winner
benchmark as well as the utility distribution.

## Direct derivation of effectiveness (no reproducing assumption needed)

For one voter write x = u − ubar*1 and b = w − mean(w)*1. Let

    q = ||b||² = k(m−k)/m,
    g = E[x · b] = E[max(A,B)],
    P = I − 11ᵀ/m.

Both possible ballots have the same q, including after approval normalization.
Candidate symmetry gives

    Cov(b)    = q/(m−1) P,
    Cov(x,b)  = g/(m−1) P,
    Cov(x)    = (1/12) P.

For many independent voters, the joint central limit theorem makes total
centered utilities and scores jointly Gaussian. Gaussian regression gives

    E[total x | total b] = (g/q) total b.

Total scores have the distribution sqrt(n*q/(m−1)) P Z for independent
standard-normal coordinates Z. Their winning index is argmax Z. Therefore

    E[utility of winner] − n/2
        ~ sqrt(n) * g/sqrt(q*(m−1)) * E[max Z],
    E[maximum utility] − n/2
        ~ sqrt(n/12) * E[max Z].

Taking the ratio yields

    effectiveness(m,k) = sqrt(12*m / ((m−1)*k*(m−k))) * E[max(A,B)].

This argument applies to any candidate-symmetric strategy using these
ballots. Since q is fixed, maximizing A or B individually also maximizes
asymptotic effectiveness. No different choice between the two ballots can
raise the last three rows to the quoted values.

## An exact five-candidate counterexample

For m=5, k=2, use ascending order statistics X1 <= ... <= X5. Then

    A = X4 + X5 − 2*ubar,
    B = 2*ubar − X1 − X2,
    D = A−B = (X1 + X2 − 4*X3 + X4 + X5)/5.

E[A] = E[B] = 1/2. The six spacings

    X1, X2−X1, X3−X2, X4−X3, X5−X4, 1−X5

are uniform on the simplex (Dirichlet(1,1,1,1,1,1)). The coefficients of D
on these spacings are 0, −1/5, −2/5, 2/5, 1/5, 0. Integration of the
positive part over that simplex gives

    E[D_+] = (1/6) * (4/15 − 1/30) = 7/180.
    E[max(A,B)] = E[B] + E[D_+] = 97/180.
    effectiveness(5,2) = (97/180)*sqrt(5/2) = 0.8520581473...

For completeness, k=1 gives (31/80)*sqrt(15/4) = 0.7503905233..., so k=2
really is best among all distinct permitted k. The quoted 86.96% is
impossible for this five-candidate system under the stated model.

### General exact integration used in the audit

For ascending utilities the coefficients a_i of D are 1−2k/m in the two
outer k blocks and −2k/m in the middle. Its coefficients on the m+1
spacings are d_j = sum_{i>j} a_i, j=0,...,m. For distinct knots d_j,

    E[D_+] = 1/(m+1) * sum_{d_i>0} d_i**(m+1) / product_{j!=i}(d_i−d_j).

Repeated knots are handled by the continuous limit, implemented as confluent
divided differences of x_+**(m+1). Finally

    E[max(A,B)] = k(m−k)/(2(m+1)) + E[D_+].

The implementation uses rational arithmetic, not numerical quadrature or
a fit to the claimed effectiveness values. Independent Sobol integration
of the original utility cube checks these exact integrals.

## Many-candidate limit

For k/m → alpha in (0,1/2], the two choices each have mean gain
k(m−k)/(2(m+1)). The extra expected gain from choosing between them is
O(sqrt(m)), hence contributes only O(1/sqrt(m)) to effectiveness. (D is a
bounded-coordinate-Lipschitz function of m independent uniforms, so its
variance is O(m), and E[D]=0.) Thus

    effectiveness → sqrt(3*alpha*(1−alpha)).

This is maximized at alpha=1/2, giving sqrt(3)/2 = **86.6025%**, not 92.25%.
Sequences with k/m→0 cannot do better: max(A,B) <= k, which bounds their
effectiveness by sqrt(12*m*k/((m−1)*(m−k))) → 0. Thus the best k/m tends
to 1/2; this is not merely a statement about fixed alpha.

## Reproducible checks

`weber_vfoak_audit.py` prints the exact results above. Optional full-election
Monte Carlo uses an independent ballot implementation, uniform utilities,
and averages over tied winners (equivalent to fair random tie-breaking).
It subtracts each realized society's mean candidate utility rather than
the constant n/2; this is unbiased variance reduction. It takes a ratio of
mean utility excesses, not a mean of election-by-election ratios.

Commands (with the repository dependencies installed):

```sh
PYTHONPATH=. python examples/weber_vfoak_audit.py
PYTHONPATH=. python examples/weber_vfoak_audit.py --elections 30000
PYTHONPATH=. python -m pytest tests/test_weber_vfoak_audit.py -q -o addopts=''
```

Seed 20261025; 30,000 elections per row; 1,001 voters:

| m | Simulated SUE (%) | Approx. 95% Monte Carlo half-width (pp) |
| ---: | ---: | ---: |
| 2 | 81.398 | 0.612 |
| 3 | 87.668 | 0.360 |
| 4 | 80.773 | 0.432 |
| 5 | 85.295 | 0.335 |
| 6 | 83.725 | 0.343 |
| 10 | 85.277 | 0.280 |

These are finite-electorate estimates, not exact asymptotic results. A 95%
interval need not contain its target on every row; the m=6 estimate is
about 2.25 standard errors below its asymptotic value. The exact integration
and independent utility-cube integration are the checks not subject to
finite-electorate convergence.

All 40 focused tests passed, including exhaustive ballot optimization,
independent utility-cube integration, reduction to vote-for-k at m=2k,
vertex enumeration, recovery of the erroneous 86.96%, exact rational
triangulation of the correct five-candidate strategy region, and exact
best-k selection for m=9,20,50,100.
The full suite was also run: 210 passed and 2 failed in unchanged code.
The existing VFOAK test still expects signed ballots, whereas this branch's
implementation returns approval-normalized ballots. The other failure is
an approval-input validation test encountering NumPy's unsigned conversion
OverflowError instead of the expected ValueError.

No historical formula implementations or library strategy code were changed:
the quoted expressions remain intact as evidence of the discrepancy. The
examples README now labels the affected historical column and links this audit.
The supplied printed scans verify the vertex list, centroid, effectiveness
formula, asymptotic claim, and table entries; the mathematical counterexample
and corrected results are independently tested.

## Follow-up review and exact best-k evaluation

After independent review, both Arena instances agree on the substantive
findings. Instance 1's follow-ups on
[PR #54](https://github.com/endolith/elsim/pull/54#issuecomment-6420444710),
[PR #57](https://github.com/endolith/elsim/pull/57#issuecomment-6420448192), and
[issue #110](https://github.com/endolith/elsim/issues/110#issuecomment-6421295966)
resolve the earlier disagreements about ballot independence, internal symmetry
checks, special-case tests, quadrature terminology, and finite-simulation bias.

The distinction between a utility-independent sign choice and a completely
utility-independent ballot matters: PR #57 retains utility information and
has the vote-for-k limiting effectiveness, while PR #54 supplies no preference
information and has zero expected efficiency. For k=m/2 the two sign choices
are equivalent, and those cases remain useful consistency tests. No universal
converse classifying every accidental agreement of the historical formula
is claimed here from finite checks alone.

The implementation in `weber_vfoak_audit.py` evaluates the finite divided-
difference formula with exact rational arithmetic for any valid integer m,k;
it does not require quadrature or geometric triangulation to get the expected
gain. Only the final effectiveness's square root is evaluated in floating
point. The `best_k(m)` helper compares the rational squared effectiveness
(up to a common factor), so its selection itself is exact.

| m | Exact best k | Corrected SUE (%) | Historical rounded k | SUE at that k (%) | Gain (pp) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 9 | 4 | 84.8517 | 3 | 83.9393 | 0.9124 |
| 20 | 9 | 85.7806 | 7 | 84.5209 | 1.2597 |
| 50 | 22 | 86.3292 | 18 | 84.6663 | 1.6629 |
| 100 | 46 | 86.4853 | 37 | 84.7891 | 1.6962 |

These larger cases are included in the tests. Their values and optimal k agree
with Instance 1's numerical results. This is independent confirmation of the
results, not a code review of the other instance's branch. The present branch
is an audit based on the existing `vfoak` implementation, not a port of the
strategy and historical-expression API onto current master.

Prepared by Arena.ai Agent Mode, **Instance 2**, with the repository owner's
source images and feedback. Links to the discussion distinguish these findings
from earlier OpenCode and Arena Instance 1 comments.
