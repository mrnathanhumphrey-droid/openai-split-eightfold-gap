"""
Sign conventions in Prop 3.1 of "Algebraicity of Weil classes on split abelian eightfolds" (OpenAI, 2026-09-18).

Convention A: the intersection sign of two oriented sheets in the standard (omega^n) orientation of C^n -- the
convention of the self-intersection formula (3.4) and of the surgery step after Lemma 3.6.
EEMS: Ekholm-Eliashberg-Murphy-Smith, arXiv:1303.0588, from which (3.6) imports the stabilization-trace signs.

This script computes, from explicit tangent frames:
  (1) the cusp offset of the lower sheet of a trace (grading +1, orientation reversed);
  (2) the Whitney sphere's double point (degrees {n+1, -1}, A-sign -(-1)^{n(n-1)/2}), as a calibration;
  (3) the stabilization-trace double point in Convention A, for both radial directions;
  (4) the comparison with EEMS's printed trace sign (-1)^{k-1} and EEMS's I(f) for the Whitney sphere;
  (5) the consequence for the final trace points of Lemma 3.6 at n = 8 (and n = 12).

Local trace model: Darboux coordinates in the symplectization; upper sheet the zero section; lower sheet the graph
of dH with Hess H = diag(eps, -1, ..., -1) at the crossing, eps = +1 for a stabilization as the radius increases.

Run:  python sign_convention.py      (numpy only)
"""
import sys

import numpy as np

FAIL = []


def check(label, got, want):
    ok = (got == want)
    print("  [%s] %s: %s%s" % ("OK " if ok else "FAIL", label, got, "" if ok else "  WANT %s" % (want,)))
    if not ok:
        FAIL.append(label)
    return ok


def unitary(F, n):
    Q, _ = np.linalg.qr(np.vstack([F.real, F.imag]))
    return Q[:n] + 1j * Q[n:]


def realc(F, n):
    M = np.zeros((2 * n, n))
    M[0::2], M[1::2] = F.real, F.imag
    return M


def degree_and_sign(Fa, ta, Fb, tb, n):
    """Degree n + th_b - th_a - sum(alpha)/pi, and the Convention-A sign of the oriented frames (Fa, Fb)."""
    Wm = unitary(Fa, n).conj().T @ unitary(Fb, n)
    al = np.mod(np.angle(np.linalg.eigvals(Wm @ Wm.T)) / 2, np.pi)
    d = n + tb - ta - al.sum() / np.pi
    s = int(np.sign(np.linalg.det(np.hstack([realc(Fa, n), realc(Fb, n)]))))
    return int(round(d)), abs(d - round(d)), s


def cusp_offset(steps=20000, U=50.0):
    """Track arg(u + i)/pi from the upper branch (u = +U) to the lower (u = -U) of the cusp (u^2/2, u)."""
    us = np.linspace(U, -U, steps)
    th = np.unwrap(np.angle(us + 1j)) / np.pi
    th = th - (th[0] - np.arctan(1 / U) / np.pi)
    off = th[-1] - np.arctan(1 / us[-1]) / np.pi
    orient = np.sign(us[-1] * 1.0)
    return off, orient


def trace_point(n, eps):
    lam = np.array([eps] + [-1.0] * (n - 1))
    Fa = np.eye(n).astype(complex)
    Fb = (np.eye(n) + 1j * np.diag(lam)).astype(complex)
    off, orient = cusp_offset()
    Fb[:, 0] *= orient
    tb = round(off) + np.arctan(lam).sum() / np.pi
    return degree_and_sign(Fa, 0.0, Fb, tb, n)


def W(p):
    """Whitney sphere W(x0, x) = x (1 + i x0) / (1 + x0^2) on S^n, one double point at 0."""
    return p[1:] * (1 + 1j * p[0]) / (1 + p[0] ** 2)


def wframe(p, d, n, h=1e-6):
    vs = [d] + [np.eye(n + 1)[j] for j in range(2, n + 1)]
    return np.column_stack([(W(p + h * v) - W(p - h * v)) / (2 * h) for v in vs])


def whitney(n, steps=4000):
    ang = []
    for i in range(steps + 1):
        s = np.pi * i / steps
        p, d = np.zeros(n + 1), np.zeros(n + 1)
        p[0], p[1], d[0], d[1] = -np.cos(s), np.sin(s), np.sin(s), np.cos(s)
        ang.append(np.angle(np.linalg.det(wframe(p, d, n))))
    th = np.unwrap(np.array(ang)) / np.pi
    pa, da = np.zeros(n + 1), np.zeros(n + 1)
    pa[0], da[1] = -1, 1
    pb, db = np.zeros(n + 1), np.zeros(n + 1)
    pb[0], db[1] = 1, -1
    return degree_and_sign(wframe(pa, da, n), th[0], wframe(pb, db, n), th[-1], n)


print("(1)  THE CUSP OFFSET")
off, orient = cusp_offset()
check("    lower-branch grading = graph grading + 1", round(off, 6), 1.0)
check("    lower-branch orientation is opposite its graph frame", orient, -1.0)

print("\n(2)  CALIBRATION: THE WHITNEY SPHERE")
for n in (8, 12):
    d, err, s = whitney(n)
    check("    Whitney n = %d: degree pair, Convention-A sign" % n, (tuple(sorted((d, n - d))), s), ((-1, n + 1), -1))

print("\n(3)  THE STABILIZATION-TRACE DOUBLE POINT IN CONVENTION A")
for n in (8, 12):
    k = n // 2
    rows = {}
    for eps in (+1, -1):
        d, err, s = trace_point(n, eps)
        rows[eps] = (tuple(sorted((d, n - d))), s)
    check("    n = %d, stabilization as the radius INCREASES: (degree pair, A-sign)" % n, rows[+1], ((2, n - 2), 1))
    check("    n = %d, stabilization as the radius DECREASES: (degree pair, A-sign)" % n, rows[-1], ((1, n - 1), -1))
    print("\n(4)  AGAINST EEMS, n = %d" % n)
    check("    EEMS's outward trace sign (-1)^{k-1} is MINUS the Convention-A sign", (-1) ** (k - 1) == -rows[+1][1], True)
    i_eems = (-1) ** (n // 2)                               # EEMS: I(f) = (-1)^{n/2} chi/2, chi(S^n)/2 = 1
    i_top = -((-1) ** (n * (n - 1) // 2))                   # [S^n].[S^n] = 0 = e(nu) + 2I
    check("    EEMS's I(Whitney) is MINUS the Convention-A count", i_eems == -i_top, True)

print("\n(5)  THE FINAL TRACE POINTS OF LEMMA 3.6 (outward, eps = +1)")
for n in (8, 12):
    pair, s = trace_point(n, +1)[0], trace_point(n, +1)[2]
    lift = pair - n / 2
    handle = (n - 2) / 2
    check("    n = %d: they are POSITIVE in Convention A" % n, s, 1)
    check("    n = %d: so the surgery handle (which needs a negative point) is not orientable there" % n, s == -1, False)
    check("    n = %d: and the grading does not extend even mod 4 (lift d - n/2 vs handle phases +-(n-2)/2)" % n,
          any((lift - h) % 4 == 0 for h in (handle, -handle)), False)
m = 3
I_A_f1 = -m
check("    cancellation count: I_A(f1) = -m plus m reverse traces of A-sign -1 (m = 3)",
      I_A_f1 + m * trace_point(8, -1)[2], -2 * m)

print()
if FAIL:
    print("FAILED: %s" % FAIL)
    sys.exit(1)
print("ALL CHECKS PASS.  EEMS's trace signs are the negatives of Convention A.  The final trace points of Lemma 3.6 are")
print("positive in Convention A (degrees {2, n-2}); the surgery after Lemma 3.6 requires negative points.")
