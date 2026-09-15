# External Theoretical Cross-Check: Quartic GW Dispersion in Hořava–Lifshitz Gravity

## 1. External paper

Araújo Filho, A. A., Silva, J. L. A., Heidari, N., Zhu, Jie, Lobo, I. P. &
Bezerra, V. B., "Gravitational wave propagation in Hořava-Lifshitz
gravity," arXiv:2607.17431 (July 2026).

The paper investigates GW generation and propagation in the leading
parity-even infrared truncation of Hořava-Lifshitz gravity, characterized
by the modified tensor dispersion relation

$$
\omega^2 = k^2 + \alpha k^4 .
$$

This is an entirely independent derivation: different authors, different
starting theory (Hořava-Lifshitz quantum gravity, not the Λ-model's
Hamiltonian construction), arriving at the same quartic spatial-dispersion
functional form.

## 2. Quartic-dispersion correspondence

In the flat-space limit, the Paper-3 propagation ansatz is

$$
\omega^2 = c^2k^2(1+\Lambda_{\rm GW}k^2) = c^2k^2 + c^2\Lambda_{\rm GW}k^4 .
$$

This has the same $k^4$-correction structure as the Hořava-Lifshitz
$\omega^2=k^2+\alpha k^4$ once the unit conventions are made explicit (Sections 3–4). The
correspondence is at the level of functional form only; it does not by
itself imply the two theories share a common physical origin for the
quartic term.

## 3. Conversion chain — kept as two independent, separately-sourced steps

It is important not to present this as a single formula lifted from the
HL paper. It is two separate steps from two separate sources, which
happen to compose.

### 3.1 Step A — HL paper's own definition (verified directly from source,

Section VI.1, "Mapping to the LVK modified-dispersion framework")

The paper states explicitly, verified against the HTML source:

> "where $[\alpha]=L^2$ and $c=\hbar=1$. To avoid confusing the
> Hořava-Lifshitz coefficient $\alpha$ with the exponent used in the LVK
> parametrization, we denote the latter by $\beta$."

and gives the LVK modified-dispersion form

$$
E^2 = k^2c^2 + A_\beta k^\beta c^\beta ,
$$

with the direct identification, in Hořava-Lifshitz natural units
($c=\hbar=1$), stated in the paper as Eq. (104):

$$
\boxed{\beta = 4, \qquad A_4 = \alpha_{\rm HL}} \qquad \text{[HL paper, natural units]} .
$$

GWTC-4.0 bound, as quoted in the source paper (its Eq. 112, 114):

$$
\bar A_4 \in [-0.62, +0.19]\times 10^3 \ {\rm eV}^{-2} \ (90\%\ {\rm cred.})
\;\Longrightarrow\;
-6.2\times10^2 < \alpha_{\rm HL} < 1.9\times10^2 \ {\rm eV}^{-2} .
$$

**Independent unit check performed here:** using $1\,{\rm eV}^{-1} =
1.97327\times10^{-7}$ m, so $1\,{\rm eV}^{-2} = 3.8938\times10^{-14}\,{\rm
m^2}$:

$$
-0.62\times10^3 \times 3.8938\times10^{-14} = -2.414\times10^{-11}\ {\rm m^2},
\qquad
+0.19\times10^3 \times 3.8938\times10^{-14} = +7.398\times10^{-12}\ {\rm m^2} .
$$

This reproduces the paper's own quoted SI interval,
$-2.41\times10^{-11}\,{\rm m^2} < \alpha_{\rm HL} < 7.40\times10^{-12}\,{\rm
m^2}$ (paper's Eq. 114), to the precision quoted — independently
recomputed here, not merely copied.

### 3.2 Step B — Paper-3 convention (separate source: this project's own,

previously-audited GWTC-4.0 cross-check in `PAPER3_VALIDATION_STATUS.md`)

$$
A_4 \;\longrightarrow\; \Lambda_{\rm GW} = \hbar^2c^3 A_4 = \frac{h^2c^3}{4\pi^2}A_4
\qquad \text{[Paper-3 convention, independent of the HL paper]} .
$$

(The two forms of the prefactor are identical: $\hbar=h/2\pi \Rightarrow
\hbar^2=h^2/4\pi^2$ — confirmed by direct substitution.)

### 3.3 Composing the two steps

Because Step A's $A_4$ bound and Step B's $A_4$ input are drawn from the
*same* published GWTC-4.0 table (both papers cite the identical LVK
parameterized-tests result), composing the two independently-sourced
steps reproduces exactly the $\Lambda_{\rm GW}$ interval already on
record in this project:

$$
\Lambda_{\rm GW} \in [-7.2\times10^{-3},\ +2.2\times10^{-3}]\ {\rm m^3/s} .
$$

No new numerical computation is introduced at this point; this is a
consistency composition of two independently-sourced conversions, not a
new measurement.

## 4. Unit and convention audit — summary

| Quantity           | Convention   | Dimension       | Value                                           | Source                                      |
| ------------------ | ------------ | --------------- | ----------------------------------------------- | ------------------------------------------- |
| $\alpha_{\rm HL}$  | $c=\hbar=1$  | $L^2$           | $=A_4$                                          | HL paper, Eq. 104                           |
| $\alpha_{\rm HL}$  | SI           | $L^2$           | $[-2.41\times10^{-11}, +7.40\times10^{-12}]$ m² | HL paper Eq. 114; reproduced here §3.1      |
| $A_4$              | LVK/GWTC-4.0 | ${\rm eV}^{-2}$ | $[-0.62,+0.19]\times10^3$                       | LVK GWTC-4.0 parameterized tests            |
| $\Lambda_{\rm GW}$ | Paper 3      | $L^3/T$         | $[-7.2\times10^{-3},+2.2\times10^{-3}]$ m³/s    | This project, `PAPER3_VALIDATION_STATUS.md` |

$\alpha_{\rm HL}$ and $\Lambda_{\rm GW}$ are **not** the same quantity and
are never set equal; both are obtained from the same underlying $A_4$ by
two different, dimensionally-appropriate, and independently-sourced
conversions (Step A from the HL paper, Step B from this project).

## 5. Additional independently-verified check: propagation-phase structure

Beyond the dispersion-relation correspondence, the HL paper derives a
propagation phase correction (its Eqs. 107–108):

$$
\delta\Psi_4^{\rm prop} \to -4\pi^3\alpha f^3 r
\quad\text{(local/flat-space limit)},
$$

which the paper writes equivalently as $-\tfrac{\alpha}{2}(2\pi f)^3 r$.

**Independently verified here:** $\tfrac12(2\pi)^3 = \tfrac12\cdot8\pi^3 =
4\pi^3$ — the two forms are algebraically identical (checked by direct
substitution, not merely asserted).

This is a a stronger cross-check than comparing the dispersion relations alone: it shows that the *same* independent derivation propagates the
$k^4$ dispersion correction through to a cubic-in-frequency propagation
phase ($\propto f^3$), which is structurally the same frequency-dependence
as the Paper-3 propagation phase $\Delta\Psi(f)\propto K(z)f^3$ (Section
5.1 correction: this is a **structural**, same-power-of-$f$ correspondence,
not a claim that the two phase formulas are numerically identical or
derived the same way).

The paper's associated amplitude correction (its Eq. 109),
$\tilde h_A \simeq \tilde h_A^{\rm GR}(1-8\pi^2\alpha f^2)e^{-4i\pi^3\alpha
f^3 r}$, is noted here for completeness but is **not** used in any claim
in this cross-check (see Section 6).

## 6. What is independently established

- An independent theoretical derivation (different authors, different
  starting theory) arrives at the same $k^4$ quartic spatial-dispersion
  structure as the Paper-3 flat-space propagation ansatz.
- That derivation is independently mapped to the standard LVK
  modified-dispersion parametrization ($\beta=4$, $A_4=\alpha_{\rm HL}$),
- providing an external check that the same parameterization is independently obtained in a distinct theoretical framework.
  
- The same independent derivation propagates the $k^4$ dispersion
  correction to a cubic-in-frequency ($f^3$) propagation phase — the same
  structural frequency-dependence used in the Paper-3 propagation phase.
- The SI unit conversion of the HL paper's own quoted bound (§3.1) and the
  propagation-phase algebraic identity (§5) were both independently
  reproduced here, not simply copied from the source.

## 7. What is NOT established

- **Not an independent observational validation.** Both this result and
  Paper 3's own GWTC-4.0 cross-check use the same GWTC-4.0 posterior
  (83 cumulative events, as the HL paper itself states); this is a
  theoretical cross-check of the dispersion structure and its
  parametrization, not a second, independent measurement.
- **Not evidence that Hořava-Lifshitz gravity and the Λ-model share a
  common physical origin.** The correspondence is at the level of
  functional form ($k^4$ dispersion, $f^3$ propagation phase) only.
- **Does not draw on the HL paper's luminosity, chirp-evolution, or
  waveform-amplitude/energy-flux derivations.** The source paper itself
  states that a full post-Newtonian constraint would additionally require
  GR/spin/waveform-systematics and conservative-dynamics inputs beyond
  its own amplitude/luminosity/chirp channel; this cross-check therefore
  uses **only** the dispersion relation (§3.1) and the propagation-phase
  structure (§5), and explicitly excludes the amplitude/luminosity/chirp
  results from any claim made here.
- **Does not establish universality of Λ** across the near-horizon
  ($\Lambda_{\rm NH}$), propagation ($\Lambda_{\rm GW}$), or analog-system
  (giant-vortex) realizations already distinguished elsewhere in Paper 3.

## 8. Claim boundary

The strongest defensible statement from this cross-check is:

> An independently-derived quartic gravitational-wave dispersion relation,
> obtained from Hořava-Lifshitz gravity by different authors using a
> different theoretical starting point, has the same functional form as
> the Λ-model's flat-space propagation ansatz, propagates to the same
> cubic-in-frequency phase structure, and maps to the identical standard
> LVK $A_4$ parametrization already used in this work's own GWTC-4.0
> cross-check. This is an external theoretical cross-check of the
> dispersion structure, its propagation-phase consequence, and its
> parametrization — not an independent observational validation (same
> GWTC-4.0 dataset), not a claim of shared physical origin between the two
> theories, and not a use of the source paper's amplitude/luminosity/chirp
> results.

## Provenance

- Dispersion relation, unit convention ($c=\hbar=1$, $[\alpha]=L^2$), the
  $\beta=4$, $A_4=\alpha$ identification, and the propagation-phase
  formulas: verified directly from the arXiv HTML source
  (arxiv.org/html/2607.17431v1), Section VI.1 and Eqs. (104), (107)–(109),
  (112), (114).
- GWTC-4.0 $A_4$ bound and its SI conversion: as quoted in the source
  paper; SI conversion and the propagation-phase algebraic identity
  ($4\pi^3 = \tfrac12(2\pi)^3$) independently re-derived here (§3.1, §5)
  and confirmed to match.
- $\Lambda_{\rm GW}$ value: reuses the existing, previously-audited
  $A_4\to\Lambda_{\rm GW}$ conversion from `PAPER3_VALIDATION_STATUS.md`
  (Step B, §3.2); no new conversion factor is introduced, and this step
  is kept explicitly separate from the HL paper's own Step A.
- No claim in this document is based on the HL paper's luminosity,
  chirp-evolution, or waveform-amplitude/energy-flux sections.
