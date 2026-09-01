# Diffusion sampler divergence: diagnosis and fix

## Symptom

The first reported evaluation gave normalized RMSE of 152 / 214 / 258 on unobserved
pixels at 30% / 60% / 90% missingness. Reconstruction panels in
`plots/phase2_diffusion/` carry colorbars spanning roughly ±600 to ±1000 against
truth fields spanning −2.5 to 1.1, with per-pixel sampling standard deviations of
400–1400. The reverse process was diverging; the `nan_to_num(posinf=1e3)` guard at
the end of `sample()` was clipping the result rather than protecting it.

Training was never the problem. The 60%-sparsity history converges cleanly
(0.29 → 0.011), and probing the trained checkpoint directly confirms the network
learned the denoising objective:

| t | MSE(ε̂, ε) | gain k = ⟨ε̂,ε⟩/⟨ε,ε⟩ | std(ε̂) |
|---|---|---|---|
| 299 | 0.0028 | 1.0009 | 1.0018 |
| 200 | 0.0020 | 0.9991 | 1.0002 |
| 100 | 0.0021 | 0.9941 | 0.9935 |
| 50 | 0.0036 | 0.9991 | 1.0009 |
| 10 | 0.0296 | 0.9601 | 0.9732 |
| 1 | 0.3648 | 0.5517 | 0.6841 |
| 0 | 1.0047 | 0.0001 | 0.0665 |

Calibration is excellent for t ≥ 50. (It collapses at t → 0 because the schedule
sets ᾱ₀ = 1 exactly, so x_t = x₀ and the ε target is unlearnable there; that is a
separate, minor defect.)

## What it was not

Two plausible hypotheses were tested against the real checkpoint and both were
wrong. Reported on 4 cases / 1 sample at 30% missing, so the scale differs from
the 20-sample evaluation figure:

| Sampler variant | RMSE (unobserved) |
|---|---|
| As committed | 316.5 |
| Shifted ᾱ index (fixes an off-by-one) | 322.2 |
| RePaint-style noised pasting of observations | 328.3 |
| Both | 333.9 |

There *is* a genuine off-by-one — the posterior uses `alphas[step]` where
α_t = ᾱ_t/ᾱ_{t−1} = `alphas[step-1]` — and pasting noiseless observations into a
partially-noised state *is* a real train/test mismatch. Neither causes the
divergence. A simulation with an exact analytic denoiser confirms the reverse
recursion is stable under both variants, and stays stable under additive ε̂ error
up to σ = 0.2.

## What it was

Tracing `std(x)` through the reverse process shows the failure in the first step:

| step | std(x) in | std(ε̂) | 1/√α | coeff2 | std(x) out |
|---|---|---|---|---|---|
| 299 | 1.00 | 1.00 | **99.99** | 1.0000 | **3.11** |
| 298 | 3.11 | 1.31 | 2.00 | 0.7500 | 4.23 |
| 275 | 24.5 | 1.35 | 1.04 | 0.0787 | 25.4 |
| 200 | 89.0 | 1.38 | 1.009 | 0.0209 | 89.8 |
| 100 | 152.5 | 1.63 | 1.003 | 0.0121 | 153.0 |
| 0 | 175.4 | 1.93 | 1.000 | 1.5640 | 172.4 |

The causal chain:

1. The cosine schedule evaluates ᾱ at t = T as cos²(π/2) = **0 exactly**. So
   β_T = 1, clipped to 0.9999, giving α_T = 10⁻⁴ and a leading coefficient
   1/√α_T ≈ **100** on the very first reverse step.
2. The model's residual (x − ε̂) at t = 299 has std ≈ 0.054 — small, but ×100
   puts x at std 5.4 when it should be ≈ 1.
3. x is now far outside the training distribution. Because the U-Net is
   GroupNorm-normalised throughout, ‖ε̂‖ saturates at roughly 1–2 **regardless of
   ‖x‖** — visible in the trace, where std(ε̂) stays near 1.3 while std(x) climbs
   past 100. The correction term is bounded; there is no restoring force.
4. The remaining steps have 1/√α slightly above 1 and a correction coefficient
   decaying toward zero, so they amplify essentially unopposed:
   ∏ 1/√α ≈ 192 over steps 298→0. Final std ≈ 175.

The 100× coefficient is harmless for a *perfect* denoiser (the residual it
multiplies is then zero) which is why the analytic-denoiser simulation stayed
stable. It is fatal for a real one.

## The fix

Use the x₀-parameterised posterior instead of the algebraically equivalent ε form,
and floor ᾱ_t away from zero. The two forms agree in exact arithmetic but not
numerically: predicting x₀ keeps every coefficient O(1) and never forms 1/√α_T.

```python
a_t   = clamp(alphas_cumprod[step + 1], min=1e-5)   # abar_t, floored
a_prev = alphas_cumprod[step]                        # abar_{t-1}
alpha  = clamp(a_t / a_prev, 1e-8, 1.0)              # exact, no off-by-one

x0_pred = (x - sqrt(1 - a_t) * eps_pred) / sqrt(a_t)
x0_pred = clamp(x0_pred, -4, 4)                      # optional thresholding
mean = (sqrt(a_prev) * (1 - alpha) / (1 - a_t)) * x0_pred \
     + (sqrt(alpha) * (1 - a_prev) / (1 - a_t)) * x
```

Isolating the two guards (30% missing, 1983):

| Guard | RMSE (unobserved) |
|---|---|
| ᾱ floor only | 0.0173 |
| x₀ clipping only | see `validated.json` |
| Both | 0.0172 |
| Neither | diverges |

The ᾱ floor is the load-bearing change; x₀ clipping is redundant once it is in
place but is cheap insurance and is standard practice.

**Result: normalized RMSE on unobserved pixels goes from 152 to 0.017.** Output
field std is 0.917 against a truth std of 0.916.

## What this exposes about the task

With the sampler fixed, the model beats a Gaussian normalized-convolution
inpaint (0.024) — but only just, and that baseline is five lines of numpy. The
reason is the masking scheme: i.i.d. per-pixel masks leave every missing cell with
observed neighbours one or two grid points away, so simple interpolation is close
to optimal and a learned prior has almost no headroom.

Baselines on the unobserved region, for reference:

| Baseline | 30% | 60% | 90% |
|---|---|---|---|
| Zero-fill (= global training mean) | 1.02 / 0.96 | 1.02 / 0.96 | 1.02 / 0.96 |
| Per-gridpoint climatology | 0.60 / 0.56 | 0.60 / 0.56 | 0.60 / 0.56 |
| Gaussian inpaint | 0.024 | 0.036 | 0.081 |

`generate_structured_mask()` in `scripts/04_diffusion_train.py` drops contiguous
blocks instead, which is both closer to how real observing systems fail (satellite
swath gaps, cloud cover, station deserts) and the regime where a generative prior
is actually needed.
