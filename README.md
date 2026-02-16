# Simple Flow

A minimal, educational, and self-contained implementation of **Flow Matching** (specifically Optimal Transport Conditional Flow Matching) using PyTorch. 

This repository aims to demystify the core concepts of Flow Matching by stripping away complex abstractions, making it easy to understand the relationship between the Vector Field, the ODE solver, and the probability path.

## 🚀 Features

- **Minimal Codebase**: The entire logic (Model, Training, Sampling) is contained in a single file (`main.py`).
- **Optimal Transport Path**: Implements the linear interpolation path $x_t = (1-t)x_0 + t x_1$, which leads to straight trajectories and stable training.
- **Visualization**: Includes scripts to visualize the learned vector field and particle trajectories.

## 📝 Mathematical Background

Flow Matching trains a neural network $v_\theta(x, t)$ to regress a target vector field $u_t(x)$ that generates a probability path $p_t(x)$ from noise distribution $p_0$ to data distribution $p_1$.

In this implementation, we use the **Optimal Transport Conditional Vector Field** defined as:

$$
\psi_t(x_0, x_1) = (1 - t) x_0 + t x_1
$$

The target vector field for this path is simply the direction from the source to the target:

$$
u_t(x | x_1, x_0) = x_1 - x_0
$$

The objective function (loss) is:

$$
\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t, q(x_1), p(x_0)} \| v_\theta(\psi_t(x_0, x_1), t) - (x_1 - x_0) \|^2
$$

