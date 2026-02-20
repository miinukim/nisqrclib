# nisqrc

A pragmatic **Qiskit** implementation of **NISQRC-style** (measure & reset) quantum reservoir computing,
inspired by *"Overcoming the coherence time barrier in quantum machine learning on temporal data"*.

Included:
- Streaming quantum reservoir (mid-circuit measurement + reset)
- Short-term memory (STM) benchmark
- Classical ESN baseline (stabilized)

## Install (editable)

```bash
pip install -e .
```

## Quick demo

```bash
nisqrc-stm-demo
# or:
python examples/stm_demo.py
```

## Notes on numerical stability

The ESN baseline is stabilized by:
- parameter validation (`leak_rate in [0,1]`)
- spectral-radius scaling via **power iteration**
- state clipping and finite-value checks

If you still see non-finite states, reduce ESN `spectral_radius` and `input_scale`.
