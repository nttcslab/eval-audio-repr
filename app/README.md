# Application-specific evaluation

Some applications use their own evaluation protocols, including specialized metrics, and their benchmarking code typically restricts the models that can be evaluated. To address this limitation and enable evaluation code to work with models available on EVAR (with wrapper implementations), we modify these applications to support a broader range of models. This subproject outlines the precise steps and codes required to integrate EVAR into each application.

## Assessing the Utility of Audio Foundation Models for Heart and Respiratory Sound Analysis

For our paper:

*[D. Niizumi, D. Takeuchi, M. Yasuda, B. T. Nguyen, Y. Ohishi, and N. Harada, "Assessing the Utility of Audio Foundation Models for Heart and Respiratory Sound Analysis," to appear at IEEE EMBC, 2025](https://arxiv.org/abs/2504.18004).*

We provide code to reproduce experiments for the tasks:

- Heart sound task: CirCor  👉 [circor](circor/README_CirCor.md).
- Heart sound task: BMD-HS  👉 [bmdhs](bmdhs/README_BMDHS.md).
- Respiratory sound task: SPRSound (SPRS) 👉  [icbhi_sprs](icbhi_sprs/README_ICBHI_SPRS.md)
- Respiratory sound task: ICBHI2017 👉  [icbhi_sprs](icbhi_sprs/README_ICBHI_SPRS.md)

Please follow the instructions in each folder.
