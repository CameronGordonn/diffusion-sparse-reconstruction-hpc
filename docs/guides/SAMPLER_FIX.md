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

## Why the headline number moved with the ensemble size

The single-sample RMSE above (316.5, 4 cases) and the committed evaluation's 152.07
(20 cases, 20 samples) are the same failure at different ensemble sizes. Drawing 8
independent samples from the pre-fix sampler and averaging progressively:

| k samples | RMSE of the mean | fitted m² + σ²/k | pure 1/√k |
|---:|---:|---:|---:|
| 1 | 316.5 | 315.0 | 316.5 |
| 2 | 233.8 | 237.9 | 223.8 |
| 4 | 186.0 | 187.8 | 158.3 |
| 8 | 162.2 | 156.9 | 111.9 |
| 20 | — | 135.0 | 70.8 |

The error does **not** fall as 1/√k — that would predict 112 at k = 8 against an
observed 162, and 71 at k = 20 against the committed 152. A least-squares fit of
RMSE² = m² + σ²/k gives a systematic floor **m = 118** and an independent part
σ = 292, and reproduces all four points to within 3%.

So the diverged fields are strongly correlated across seeds, not independent noise:
the runaway is driven deterministically by the same conditioning (y, M) and the same
‖ε̂‖ saturation, so every seed lands somewhere near the same wrong answer. The
per-sample RMSEs bear this out, spanning only 290.8–316.5 across 8 draws. Ensemble
averaging removes the σ/√k part and stalls at m ≈ 118.

Extrapolating to k = 20 gives 135 against the reported 152 — reasonable agreement
given a different case set, case count and mask seed, and far closer than the 71 that
independence would imply.

The practical lesson: **a diverging sampler's reported RMSE is partly an artifact of
`num_samples`.** Any comparison across runs with different ensemble sizes was
measuring ensemble size as much as reconstruction quality. Sensitivity of the
headline metric to `num_samples` is worth checking as a routine diagnostic — a
metric that moves substantially with it is a sign that sampling, not the model, is
setting the score.

## The fix

The repair is a **guard on the schedule endpoint, not a change of
parameterisation**. Substituting x̂₀ = (x − √(1−ᾱ)ε̂)/√ᾱ into the x₀-form
posterior mean recovers the ε form exactly, so the parameterisation cannot by
itself do anything — and the ablation below confirms it, since the ungated x₀
form diverges harder than the ungated ε form did.

Two independent guards each remove the singularity:

- **Floor ᾱ_t** (`alpha_cumprod_min = 1e-5`). Because α_t is formed as
  ᾱ_t / ᾱ_{t−1} rather than read from the pre-clipped β array, the floor
  propagates into α: at t = T it gives α = 1e-5 / 2.70e-5 = 0.371 and
  1/√α = **1.64** instead of 100.
- **Clamp x̂₀** to a data range (`x0_clip = 4.0`, the Improved-DDPM convention).
  This bounds the posterior mean however large 1/√α becomes.

The x₀ form is used because `x0_clip` is only naturally expressible in it, and
because it makes the floor's effect on α explicit. Both guards are on by default.

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
| x₀ clipping only | 0.0173 |
| Both | 0.0173 |
| Neither | 1.23 × 10⁶ |

Either guard alone is sufficient, and they are not additive — both address the
same singularity, which is √ᾱ_T = 0 appearing in a denominator. With neither, the
x₀ form diverges harder than the ε form did, because 1/√ᾱ_T is ~10¹⁶ rather than
~10². Keeping both is cheap insurance and matches standard practice.

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
blocks instead, which is closer to how real observing systems fail (satellite
swath gaps, cloud cover, station deserts). At the same 30% missingness, block-8
masks make the interpolation baseline **6.8× harder**:

| Mask | Missing | Gaussian inpaint RMSE |
|---|---|---|
| i.i.d. pixel | 0.30 | 0.022 |
| block-8 | 0.30 | 0.151 |

**This is motivation, not yet a result.** The checkpoints in this repository were
trained on i.i.d. pixel masks, so evaluating them on block masks is
out-of-distribution for the conditioning input. A poor diffusion score under block
masking would therefore say nothing about the reconstruction problem's difficulty —
only that the conditioner was never shown gaps of that shape. What the table above
establishes is narrower and still worth stating: **structured masking breaks the
trivial baseline.** Whether a pixel-mask-trained diffusion model transfers to
structured gaps is open, and the honest way to close it is to retrain one model at
one sparsity level on block masks and re-run the comparison. That is a single
training job.

### The margin over interpolation, measured

Running the fixed sampler across both test years and all three sparsities, against
the Gaussian inpaint computed on the identical cases and masks. These are CPU
spot-check numbers (8 cases, mean of 2 samples) pending a cluster re-run, but the
two years agree closely and the pattern is unambiguous:

| Missing | Diffusion | Gaussian inpaint | Ratio |
|---|---:|---:|---|
| 30% | 0.0152 | 0.0247 | 1.62× better |
| 60% | 0.0182 | 0.0344 | 1.89× better |
| 90% | 0.1278 | 0.0815 | **1.57× worse** |

**The expected story is backwards.** The intuition — that a generative prior earns
its keep where data is scarcest — predicts the diffusion advantage should widen with
sparsity. It does the opposite, and at 90% the model loses to a five-line Gaussian
smoother. Both years agree (1983: 0.119 vs 0.078; 1985: 0.137 vs 0.085), so this is
not a seed artifact.

Two readings, not yet separated: the 90% model may simply be undertrained on a
much harder objective, or the learned prior may be weak enough that it only helps
while the conditioning is dense enough to carry most of the answer. Distinguishing
them means looking at the 90% training curve and at whether the model beats
climatology in the interior of large gaps, which is future work.

What this does settle is that **there is no sparsity level in this setup where
diffusion is compellingly better than interpolation.** With i.i.d. pixel masks the
best case is a 1.9× margin over five lines of numpy. That is the strongest argument
for the structured-mask experiment above: it is the remaining setting where a
learned prior has a plausible reason to win, and it is now one job away.
