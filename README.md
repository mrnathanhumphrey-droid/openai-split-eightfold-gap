# openai-split-eightfold-gap

> **Status.** Published 2026-10-07 (18:17 UTC) and reported to OpenAI the same afternoon. On 2026-10-08 (05:18 UTC)
> OpenAI withdrew this paper and two papers that depend on it, citing "a sign error [that] invalidates a
> stabilization-trace cancellation argument" ([history](https://github.com/openai/math/blob/main/history.md)).
> Worked out by Nathan Humphrey with Claude and a GPT model.

A sign-convention check of Proposition 3.1 in *Algebraicity of Weil classes on split abelian eightfolds* (OpenAI, 2026-09-18):
https://github.com/openai/math/blob/main/preprints/Algebraicity-of-Weil-classes-on-split-abelian-eightfolds-September-18-2026/paper.pdf

## What the proof of Prop 3.1 does

- (3.4) counts the double points of the immersion with the self-intersection formula. That count uses the standard orientation of Cⁿ; call it Convention A.
- (3.6) takes the signs of the stabilization-trace double points from Ekholm–Eliashberg–Murphy–Smith (EEMS), arXiv:1303.0588.
- Lemma 3.6 inserts reverse traces to cancel the double-point count. It then inserts forward traces and fills the center with the real plane.
- The surgery that follows needs the remaining double points to be negative.

## What the scripts compute

- EEMS's trace signs, and EEMS's double-point count for the Whitney sphere, are the negatives of Convention A.
- In Convention A, the forward-trace double points left by Lemma 3.6 are **positive**, with degrees {2, n−2}. At those points the surgery handle is not orientable, and the grading does not extend even mod 4.
- Read in a single convention, the cancellation count I(f₁) + m comes out as −2m, not 0.
- In every even dimension n = 2..14, the forward-trace double point is not orientably surgerable.
  - This is shown two ways, which agree: a χ-count, and Polterovich's surgery criterion as quoted in EEMS.
  - Both are calibrated on the Whitney sphere.
  - At n = 2 this matches the classical fact that a stabilized Legendrian unknot has no orientable Lagrangian filling.

This concerns the construction in the proof of Prop 3.1. It does not show that the paper's theorem is false.

## Run

```
pip install numpy
python sign_convention.py
python every_even_dimension.py
```

Each script prints its checks and exits 0 if all of them pass.
