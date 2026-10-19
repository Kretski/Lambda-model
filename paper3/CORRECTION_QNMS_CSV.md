# Correction: what qnms.csv contains (October 2026)

The EXP B file qnms.csv, provided by P. Švančara (arXiv:2502.11209), contains the QNM frequencies
**predicted by the authors' model** (resonances of their resonance condition, Methods Eq. 17), not
peak frequencies measured in the spectra (confirmed by P. Švančara, 19 October 2026). Measured peaks
are not tabulated by the authors, because not all QNMs are excited and the drive adds extra peaks.

Consequences for this repository:
- Every comparison labelled "experimental q=2", "experimental frequencies" or "parameter-free
  correspondence with experiment" is a comparison between two independent implementations of the
  same WKB model: the authors' and this repository's. The agreement (0.01-0.06% for |m| >= 10,
  growing to about 1% at m = -4) measures implementation agreement, not agreement with data.
- The small-|m| growth is a difference between the two implementations, not between model and nature.
- The flow parameters (C, Omega, h0) were fitted by the authors to the wave spectra (Nature 628, 66
  (2024), Fig. 2f), not to the QNMs; no parameter was fitted to the QNM values.
- Files under _archive/ and Submissions/ are kept unchanged as historical records; this note
  supersedes their wording on this point.
- The EXP C prediction (paper3/expC_prediction, tag expC-predictions-frozen) is a test of the
  implementation against the authors' model values for new parameters; a comparison with measured
  spectra has not been made.
