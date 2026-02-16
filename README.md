# simple flow

A minimal, educational, and self-contained implementation of **Flow Matching** (specifically Optimal Transport Conditional Flow Matching) using PyTorch. 

This repository aims to demystify the core concepts of Flow Matching by stripping away complex abstractions, making it easy to understand the relationship between the Vector Field, the ODE solver, and the probability path.

## 🚀 Features

- **Minimal Codebase**: The entire logic (Model, Training, Sampling) is contained in a single file (`simple_flow_matching.py`).
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

## 📦 Installation

```bash
git clone <this repository>
cd simple-flow
pip install torch numpy matplotlib scikit-learn
```

## 🏃 Usage
Simply run the script to train the model on the "Two Moons" dataset and visualize the results.

```Bash
python main.py
```

After training, the script will display:
1. Training Loss: Convergence curve.
2. Generated Samples: Mapping from Gaussian noise to the data distribution.
3. Flow Trajectories: The paths taken by particles during the ODE solving process.

## 📂 Code Structure
- VectorFieldNet: A simple MLP that takes (x,t) as input and outputs velocity vector v.
- train_flow_matching: Implements the OT-CFM loss.
- sample_flow_matching: Solves the ODE $d X_t = v_\theta(X_t, t) dt$ using Euler's method.

## 📚 References
- Flow Matching for Generative Modeling (Lipman et al., 2023)
- Flow Straight and Fast: Learning to Generate with Rectified Flow (Liu et al., 2023)

## 📄 License
MIT License