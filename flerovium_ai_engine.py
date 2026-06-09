import torch
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================================
# MODULE 1: THE QUANTUM PHYSICS ENVIRONMENT (Reaction Kinematics)
# =====================================================================
class QuantumFleroviumEnv:
    def __init__(self):
        # Reaction Parameters: Calcium-48 + Plutonium-244 -> Flerovium
        self.A_proj = 48.0      # Projectile Mass (Ca-48)
        self.Z_proj = 20.0      # Projectile Charge
        self.A_targ = 244.0     # Target Mass (Pu-244)
        self.Z_targ = 94.0      # Target Charge
        self.Q_value = -160.0   # Q-value of reaction (MeV)
        
        # Physics Constants
        self.e_sq = 1.44        # Coulomb constant
        self.R_0 = 1.25         # Nuclear radius constant (fm)

    def get_coulomb_barrier(self):
        """Calculates the physical barrier the beam must overcome."""
        R1 = self.R_0 * (self.A_proj ** (1.0/3.0))
        R2 = self.R_0 * (self.A_targ ** (1.0/3.0))
        barrier = (self.e_sq * self.Z_proj * self.Z_targ) / (R1 + R2)
        return barrier

    def calculate_quantum_loss(self, energy_per_nucleon):
        """AI Loss Function: Teaches the agent to find the Cold Fusion Sweet Spot."""
        # 1. Kinematics
        E_lab = energy_per_nucleon * self.A_proj
        E_cm = E_lab * (self.A_targ / (self.A_proj + self.A_targ))
        E_star = E_cm + self.Q_value # Excitation Energy
        
        # 2. Constraints & Penalties
        barrier = self.get_coulomb_barrier()
        target_E_star = 35.0 # We want exactly 35 MeV for Cold Fusion
        
        # Penalty 1: If energy is too low to fuse (bounce off)
        barrier_penalty = torch.relu(barrier - E_lab) * 1000.0 
        
        # Penalty 2: If the resulting atom is too hot (fission/decay) or too cold
        excitation_loss = torch.pow(E_star - target_E_star, 2) * 50.0
        
        # Total Loss (AI must minimize this to 0)
        return barrier_penalty + excitation_loss, E_lab, E_star

# =====================================================================
# MODULE 2: THE AI AGENT (Reinforcement Learning Optimizer)
# =====================================================================
print("Initializing Quantum-Informed GCRL Agent...")
env = QuantumFleroviumEnv()

# AI starts with a random guess for Beam Energy (MeV per nucleon)
predicted_energy = torch.tensor([10.0], requires_grad=True)
optimizer = optim.Adam([predicted_energy], lr=0.05)

epochs = 1500
history = []

print("Training Agent to discover Optimal Cold Fusion Parameters...")
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # AI tests its current guess in the physics environment
    loss, E_lab, E_star = env.calculate_quantum_loss(predicted_energy)
    
    # Backpropagation: AI learns and adjusts the energy
    loss.backward()
    optimizer.step()
    
    # Log progress
    if epoch % 100 == 0:
        history.append({
            'Epoch': epoch,
            'Energy_per_Nucleon': predicted_energy.item(),
            'E_lab': E_lab.item(),
            'Excitation_Energy': E_star.item(),
            'Loss': loss.item()
        })

# =====================================================================
# MODULE 3: RESULTS & VALIDATION
# =====================================================================
final_energy = predicted_energy.item()
print("\n--- AI OPTIMIZATION COMPLETE ---")
print(f"Optimal Beam Energy (per nucleon): {final_energy:.4f} MeV")
print(f"Total Beam Energy (E_lab): {final_energy * 48.0:.2f} MeV")
print(f"Final Excitation Energy (E*): {(final_energy * 48.0 * (244.0/292.0)) - 160.0:.2f} MeV")
print(f"Coulomb Barrier Cleared: {env.get_coulomb_barrier():.2f} MeV")
