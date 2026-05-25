# 🧮 Quantum Counting in Qiskit
How many solutions are in your search space? Instead of searching for them all, just count them. This project implements the Quantum Counting algorithm, combining Grover’s Search with Quantum Phase Estimation (QPE) to statistically estimate the number of marked items in a database.
# ⚡ Features
* Custom Grover Suite: Built-from-scratch implementation of the Oracle (diagonal phase) and Diffuser ($2|s\rangle\langle s| - I$).
* Manual IQFT: The Inverse Quantum Fourier Transform is implemented using textbook gate decompositions (Swaps, Hadamards, and Controlled-Phase rotations) rather than library shortcuts.
* Precision QPE: Maps the Grover operator's eigenphases to a counting register to solve for $M$ (number of solutions).
* Real-World Noise: Compare results from an ideal Aer simulator against a FakeBackend (e.g., FakeTorino) to see how decoherence impacts quantum precision.
# 🛠️ The Math Behind the Magic
The Grover operator $G$ rotates the state vector by an angle $\theta$. By estimating this phase, we recover $M$ via: 
$$\sin^2\left(\frac{\theta}{2}\right) = \frac{M}{N}$$
# 📊 Quick Start
1. Requirements:
   pip install qiskit qiskit-aer matplotlib numpy
2. Run the Experiment
   The script performs a default search on $n=3$ qubits with $M=2$ marked states (011 and 101).
   python quantum_counting_project_group1.py
# 📈 Visualizing the Phase
The script generates a quantum_counting_distributions.png file, providing a side-by-side comparison of:
* Ideal Probabilities: Sharp peaks at the bitstrings corresponding to the true phase.
* Noisy Probabilities: The "smeared" reality of running on near-term hardware (NISQ).
# 🏗️ Project Structure
* phase_oracle_unitary: Marks specific bitstrings with a $-1$ phase.
* inverse_qft_circuit: Hand-coded Fourier reversal.estimate_M_from_bitstring:
* Classical post-processing to convert binary phases into an integer count.
* Note: For higher precision, increase the COUNT_QUBITS variable to sharpen the phase estimation!
