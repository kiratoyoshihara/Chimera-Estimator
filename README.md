# The Chimera Estimator

**Stable gradients for empirical KL divergence in high dimensions.**

Schulman's `k₃` estimator is unbiased, non-negative, and low variance — but its *gradient* variance explodes exponentially with dimensionality. The Chimera Estimator uses `k₃` for the forward pass and `k₁` for the backward pass via a simple stop-gradient trick, getting the best of both.

📝 **Blog post:** [kiratoyoshihara.github.io/essays/chimera-kl.html](https://kiratoyoshihara.github.io/essays/chimera-kl.html)

## The idea

```python
loss_k1 = log_r.mean()
loss_k3 = (r - 1.0 - log_r).mean()
loss = loss_k3.detach() + loss_k1 - loss_k1.detach()
```

- **Forward value** = `k₃` (non-negative, unbiased, low variance)
- **Gradient** = `∇k₁` (no `r` factor, stable across dimensions)

## Results

At `D=5000`, the gradient variance of `k₃` reaches ~10¹⁴ while Chimera stays at ~0.03:

| D | Var(k₁) | Var(k₃) | Var(Chimera) |
|------:|--------:|-----------------:|------------:|
| 1 | 0.0317 | 0.0002 | 0.0317 |
| 100 | 0.0311 | 0.0167 | 0.0311 |
| 1000 | 0.0312 | 39.04 | 0.0312 |
| 5000 | 0.0312 | 1.56 × 10¹⁴ | 0.0312 |

## Reproducing

```bash
pip install torch matplotlib numpy
python run_variance_experiment.py
python run_training_experiment.py
```

## Related work

This post builds on [John Schulman's "Approximating KL Divergence" (2020)](http://joschu.net/blog/kl-approx.html). The observation that `k₃`'s gradient properties are problematic has been independently noted by:

- [Huang et al. (2025)](https://arxiv.org/abs/2510.01555) — gradient-centric analysis of KL regularization in RLHF
- [Paischer et al. (2025)](https://arxiv.org/abs/2512.21852) — `k₃` in reward shaping causing training collapse
- [Wang (2025)](https://xihuai18.github.io/reinforcement-learning/2025/12/01/kl-estimators-en.html) — gradient correctness across estimators

## Citation

```bibtex
@misc{yoshihara2026chimera,
  title={The Chimera Estimator: Fixing KL Divergence Gradients in High Dimensions},
  author={Kirato Yoshihara},
  year={2026},
  url={https://kiratoyoshihara.github.io/essays/chimera-kl.html}
}
```

## Author

[Kirato Yoshihara](https://kiratoyoshihara.github.io/) — Undergraduate, The University of Osaka