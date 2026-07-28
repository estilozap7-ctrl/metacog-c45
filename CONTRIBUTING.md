# Contributing Guidelines — MetaCog-C45

Thank you for your interest in contributing to **MetaCog-C45**, a scientific machine learning framework designed for metacognitive decision tree research developed at **Universidad de Córdoba** (Montería, Colombia) by **Luis Alberto Buelvas Cogollo**.

---

## Institutional & Scientific Framework Context

- **Framework:** MetaCog-C45 Framework (Version 1.0)
- **Author:** Luis Alberto Buelvas Cogollo
- **Affiliation:** Universidad de Córdoba, Montería, Colombia
- **Academic Program:** Maestría en Inteligencia Artificial e Ingeniería del Conocimiento
- **Research Domain:** Machine Learning, Decision Trees, Metacognition, Explainable AI (XAI)

---

## Principles for Contributions

MetaCog-C45 is a publication-grade scientific software artifact. To preserve scientific integrity and empirical reproducibility, all contributions must adhere to the following principles:

1. **Deterministic Reproducibility:** Any modifications to pipelines, algorithms, or utility modules must pass exact equivalence checks (`pytest tests/test_equivalence.py`).
2. **Backward Compatibility:** `mode='classic'` must remain 100% compliant with standard C4.5 behavior (Information Gain, Gain Ratio, Reduced Error Pruning).
3. **No Unjustified Algorithm Mutations:** Mathematical formulations (Split Confidence Score, Node Stability, Threshold Survival, Competitiveness Index) are grounded in theoretical models and cannot be altered without accompanying peer-reviewed empirical justification.

---

## How to Contribute

### 1. Reporting Issues
- Use GitHub Issues to report bugs or request features.
- Include Python version, OS details, step-by-step reproduction steps, and full tracebacks.

### 2. Submitting Pull Requests
- Fork the repository and create a feature branch (`git checkout -b feature/amazing-feature`).
- Ensure all tests pass:
  ```bash
  python -m pytest tests/
  ```
- Follow PEP 8 style guidelines.
- Submit a PR with a concise description of changes and scientific rationale.

---

## Code of Conduct

Please note that this project is governed by the [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). By participating, you are expected to uphold these standards.

---

## Copyright & License

By contributing to MetaCog-C45, you agree that your contributions will be licensed under the project's [MIT License](LICENSE), copyrighted © 2026 **Luis Alberto Buelvas Cogollo**.
