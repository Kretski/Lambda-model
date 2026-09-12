import numpy as np

ROOTS = {
    -14: (9.3995509552, -0.3086180377),
    -13: (9.12189,      -0.31734),
    -12: (8.83788,      -0.32801),
    -11: (8.54558,      -0.34110),
    -10: (8.2426577969, -0.3571800363),
}

EXP_Q2 = {
    -14: 9.599296405,
    -13: 9.342201757,
    -12: 9.082091798,
    -11: 8.811238183,
    -10: 8.533427811,
}


def fit_report(name, x, y):
    print()
    print("=" * 90)
    print(name)
    print("=" * 90)

    for degree in (1, 2):
        coeff = np.polyfit(x, y, degree)
        pred = np.polyval(coeff, x)
        err = y - pred

        print(f"\n  degree {degree}:")
        print(f"    MAE  = {np.mean(np.abs(err)):.10f}")
        print(f"    RMSE = {np.sqrt(np.mean(err**2)):.10f}")
        print(f"    MAX  = {np.max(np.abs(err)):.10f}")

        print("    residuals:")
        for m, e in zip(x, err):
            print(f"      m={int(m):3d}: {e:+.10f}")


def difference_report(name, x, y):
    print()
    print("=" * 90)
    print(name + " -- DIFFERENCES")
    print("=" * 90)

    d1 = np.diff(y)
    d2 = np.diff(y, n=2)

    print("\n  first differences:")
    for i, v in enumerate(d1):
        print(f"    {int(x[i]):3d} -> {int(x[i+1]):3d}: {v:+.10f}")

    print("\n  second differences:")
    for i, v in enumerate(d2):
        print(f"    centered at m={int(x[i+1]):3d}: {v:+.10f}")

    print(f"\n  max |first difference|  = {np.max(np.abs(d1)):.10f}")
    print(f"  max |second difference| = {np.max(np.abs(d2)):.10f}")


def main():
    ms = np.array(sorted(ROOTS.keys()), dtype=float)
    re = np.array([ROOTS[int(m)][0] for m in ms])
    im = np.array([ROOTS[int(m)][1] for m in ms])

    print("=" * 90)
    print("STAGE 5I-15 -- DEEP BRANCH SMOOTHNESS TEST")
    print("=" * 90)

    print("\nINPUT ROOTS")
    for m, r, i in zip(ms, re, im):
        print(f"  m={int(m):3d}  Re={r:.10f}  Im={i:.10f}")

    fit_report("Re(m) FIT", ms, re)
    fit_report("Im(m) FIT", ms, im)

    difference_report("Re(m)", ms, re)
    difference_report("Im(m)", ms, im)

    print()
    print("=" * 90)
    print("SEPARATION FROM EXPECTED q=2 BRANCH")
    print("=" * 90)

    q2_sep = []

    for m, r in zip(ms, re):
        q2 = EXP_Q2[int(m)]
        delta = r - q2
        q2_sep.append(delta)

        print(
            f"  m={int(m):3d}  "
            f"deep Re={r:.10f}  "
            f"q2 Re={q2:.10f}  "
            f"Delta={delta:+.10f}"
        )

    q2_sep = np.array(q2_sep)

    print("\n  mean |Delta Re| =", f"{np.mean(np.abs(q2_sep)):.10f}", "Hz")
    print("  min  |Delta Re| =", f"{np.min(np.abs(q2_sep)):.10f}", "Hz")
    print("  max  |Delta Re| =", f"{np.max(np.abs(q2_sep)):.10f}", "Hz")

    # Quadratic fits
    re_c = np.polyfit(ms, re, 2)
    im_c = np.polyfit(ms, im, 2)

    re_err = re - np.polyval(re_c, ms)
    im_err = im - np.polyval(im_c, ms)

    re_rmse = np.sqrt(np.mean(re_err**2))
    im_rmse = np.sqrt(np.mean(im_err**2))

    mean_sep = np.mean(np.abs(q2_sep))

    print()
    print("=" * 90)
    print("FINAL DIAGNOSTIC")
    print("=" * 90)

    print(f"\n  Re quadratic RMSE = {re_rmse:.10f} Hz")
    print(f"  Im quadratic RMSE = {im_rmse:.10f} Hz")
    print(f"  q=2 mean separation = {mean_sep:.10f} Hz")

    smooth_re = re_rmse < 0.005
    smooth_im = im_rmse < 0.002
    distinct = mean_sep > 0.05

    print("\n  smooth Re(m):       ", "YES" if smooth_re else "NO")
    print("  smooth Im(m):       ", "YES" if smooth_im else "NO")
    print("  distinct from q=2: ", "YES" if distinct else "NO")

    print()
    print("=" * 90)

    if smooth_re and smooth_im and distinct:
        print("RESULT: CONSISTENT WITH A DISTINCT SMOOTH DEEP BRANCH")
        print()
        print("The five points m=-10..-14 form a numerically smooth")
        print("sequence and remain separated from the expected q=2 branch.")
        print()
        print("This is evidence for a distinct numerical resonance")
        print("structure, NOT yet a claim of new physics.")
    else:
        print("RESULT: SMOOTHNESS / DISTINCTNESS NOT SUFFICIENTLY ESTABLISHED")

    print("=" * 90)
    print("NO q LABEL ASSIGNED")


if __name__ == "__main__":
    main()
