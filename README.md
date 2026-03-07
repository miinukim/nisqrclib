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
