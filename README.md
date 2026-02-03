# PD-OFM: Physics-Driven Orthogonal Feature Method

This repository implements the **Physics-Driven Orthogonal Feature Method (PD-OFM)**, a framework for solving Partial Differential Equations (PDEs) using physics-informed deep learning with orthogonality regularization.

## Overview
PD-OFM addresses the limitations of standard Random Feature Methods (RFM) and Physics-Informed Neural Networks (PINNs) by learning a set of nearly orthogonal basis functions tailored to the differential operator and domain geometry.

Key features:
- **Orthogonality Regularization**: Ensures diverse and non-redundant features.
- **High Accuracy**: Achieves significantly lower residual errors (2-3 orders of magnitude improvement).
- **Transferability**: Pretrained features generalize across different source terms and geometries (e.g., Square to L-shape/Annulus).

## Methodology
The method combines:
1.  **Physics-Informed Pretraining**: Minimizing PDE residuals + Orthogonality loss ($\|U^\top U - I\|_F^2$).
2.  **Least-Squares Solver**: Solving for the final linear coefficients $c^*$ directly.

## Citation
If you use this code, please cite the associated paper:
Jia, Q., & Wang, D. (2026). Physics informed learning of orthogonal features with applications in solving partial differential equations.
