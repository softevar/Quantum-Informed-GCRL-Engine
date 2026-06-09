# ⚛️ Quantum-Informed AI Engine for Superheavy Element Synthesis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-AI%20Optimizer-orange)
![Physics](https://img.shields.io/badge/Domain-Nuclear%20Kinematics-success)

## 📌 Overview
This repository contains a **Geometric-Constrained Reinforcement Learning (GCRL)** framework designed to simulate and optimize the reaction kinematics for synthesizing the superheavy element **Flerovium-298 ($Z=114, N=184$)**. 

Traditional trial-and-error methods in heavy-ion accelerators are highly resource-intensive. This AI-driven engine provides a predictive mathematical roadmap by finding the exact "Sweet Spot" that overcomes the Coulomb barrier while maintaining a "Cold Fusion" regime to prevent immediate fission.

## 🔬 The Physics Model
The simulation models the fusion of **Calcium-48** (projectile) and **Plutonium-244** (target). The AI agent's loss function is strictly constrained by real-world quantum mechanics and reaction kinematics:

1. **Coulomb Barrier ($V_c$):** The beam energy must be sufficient to overcome electrostatic repulsion.
2. **Excitation Energy ($E^*$):** The compound nucleus must remain within the Cold Fusion energy window ($\approx 35 \text{ MeV}$) to survive.

The kinematic relationships optimized by the agent are:
$$E_{cm} = E_{lab} \times \left( \frac{A_{targ}}{A_{proj} + A_{targ}} \right)$$
$$E^* = E_{cm} + Q_{value}$$

## 🚀 Key Results & Convergence
After training the agent to minimize the loss function combining barrier penetration penalties and excitation variance, the system successfully converged on the following optimal parameters:

* **Optimal Beam Energy:** `4.8770 MeV/nucleon`
* **Total Beam Energy ($E_{lab}$):** `234.10 MeV`
* **Coulomb Barrier Cleared:** `219.14 MeV`
* **Final Excitation Energy ($E^*$):** `35.00 MeV` (Perfect Cold Fusion state)

These results indicate a highly stable theoretical synthesis pathway, minimizing radiation/decay probabilities.

## 🛠️ How to Run
1. Clone this repository:
   ```bash
   git clone [https://github.com/softevar/GCRL-Superheavy-Element-Synthesis.git](https://github.com/softevar/GCRL-Superheavy-Element-Synthesis.git)
