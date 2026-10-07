"""
Orientable Lagrangian surgery at a double point, in every even dimension n = 2..14.

Chi-count.  For an oriented Lagrangian immersion f : L^n -> C^n, n even, with transverse double points,
[L].[L] = e(nu) + 2 sum s_p, with e(nu) = (-1)^{n(n-1)/2} chi(L) and s_p in Convention A (the omega^n orientation).
Resolving one point p by Lagrangian surgery lowers chi by 2 and leaves [L] unchanged; if the result is oriented the
formula holds for it with p removed.  Subtracting:  orientable surgery at p  ==>  s_p = -(-1)^{n(n-1)/2}.

Polterovich's criterion, as quoted in EEMS (arXiv:1303.0588, Section 1): for n even the surgery gives L # Q_eps with
eps = (-1)^{n(n-1)/2+1} sign(p), Q_+ = S^1 x S^{n-1}.  Read in Convention A it is the same condition.

This script checks both criteria on the Whitney sphere, computes the stabilization-trace sign in every even
dimension from explicit frames, and checks n = 2 against the classical fact that an orientable Lagrangian filling F of
a Legendrian knot in S^3 has tb = -chi(F) (so a stabilized unknot has none).

Run:  python every_even_dimension.py      (numpy only)
"""
import sys

import numpy as np

FAIL = []
EVEN = (2, 4, 6, 8, 10, 12, 14)


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
    Wm = unitary(Fa, n).conj().T @ unitary(Fb, n)
    al = np.mod(np.angle(np.linalg.eigvals(Wm @ Wm.T)) / 2, np.pi)
    d = n + tb - ta - al.sum() / np.pi
    s = int(np.sign(np.linalg.det(np.hstack([realc(Fa, n), realc(Fb, n)]))))
    return int(round(d)), abs(d - round(d)), s


def cusp_offset(steps=20000, U=50.0):
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


def e_sign(n):
    return (-1) ** (n * (n - 1) // 2)


def orientable_chi(s, n):
    return s == -e_sign(n)


def orientable_polterovich(s, n):
    return (-1) ** (n * (n - 1) // 2 + 1) * s == 1


print("(1)  THE TWO CRITERIA, CALIBRATED ON THE WHITNEY SPHERE")
WH = {n: whitney(n) for n in EVEN}
check("    Whitney: every computed degree is an integer", all(WH[n][1] < 1e-6 for n in EVEN), True)
check("    Whitney A-sign = -(-1)^{n(n-1)/2}", [WH[n][2] for n in EVEN], [-e_sign(n) for n in EVEN])
check("    chi-count: Whitney surgery orientable at every even n", [orientable_chi(WH[n][2], n) for n in EVEN],
      [True] * len(EVEN))
check("    Polterovich (Convention A): Whitney surgery gives Q_+ = S^1 x S^{n-1} at every even n",
      [orientable_polterovich(WH[n][2], n) for n in EVEN], [True] * len(EVEN))
check("    the two criteria are the same function of (sign, n)",
      all(orientable_chi(s, n) == orientable_polterovich(s, n) for s in (1, -1) for n in EVEN), True)
check("    EEMS's I(Whitney) = (-1)^{n/2} is MINUS the A-sign at every even n",
      [(-1) ** (n // 2) == -WH[n][2] for n in EVEN], [True] * len(EVEN))

print("\n(2)  THE STABILIZATION-TRACE DOUBLE POINT, n = 2 .. 14")
TR = {n: {eps: trace_point(n, eps) for eps in (1, -1)} for n in EVEN}
check("    every trace degree is an integer", all(TR[n][e][1] < 1e-6 for n in EVEN for e in (1, -1)), True)
check("    outward (forward) trace A-sign = (-1)^{n/2}", [TR[n][1][2] for n in EVEN], [(-1) ** (n // 2) for n in EVEN])
check("    inward (reverse) trace A-sign = minus the outward one", [TR[n][-1][2] == -TR[n][1][2] for n in EVEN],
      [True] * len(EVEN))
check("    the forward trace point is orientably surgerable in NO even dimension",
      [orientable_chi(TR[n][1][2], n) for n in EVEN], [False] * len(EVEN))
check("    reason: n(n-1)/2 = n/2 mod 2, so (-1)^{n/2} = (-1)^{n(n-1)/2} always",
      [(n * (n - 1) // 2 - n // 2) % 2 for n in EVEN], [0] * len(EVEN))

print("\n(3)  n = 2 AGAINST THE CLASSICAL FACT")
CHI_NEEDED = {r: 1 + 2 * r for r in (1, 2, 3)}
check("    an embedded filling replacing 'disk + r forward traces' needs chi(F) = 1 + 2r; a compact orientable surface with"
      " one boundary circle has chi <= 1", [CHI_NEEDED[r] <= 1 for r in (1, 2, 3)], [False, False, False])
check("    and the trace computation at n = 2 agrees: not orientably surgerable", orientable_chi(TR[2][1][2], 2), False)

print()
if FAIL:
    print("FAILED: %s" % FAIL)
    sys.exit(1)
print("ALL CHECKS PASS.  The forward stabilization-trace double point is not orientably surgerable in any even")
print("dimension n = 2..14 (chi-count and Polterovich agree, calibrated on the Whitney sphere).")
