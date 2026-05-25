"""
Quantum Counting in Qiskit

This script implements:

1. Build a Grover oracle for a Boolean function with known solutions.
2. Build the Grover iteration G = D O on the search register.
3. Apply Quantum Phase Estimation (QPE) to G.
4. Implement the inverse QFT from scratch.
5. Compare ideal counts with noisy counts from a FakeBackend.
6. Recover the number of marked items M from the measured phase.


Notes
- The Grover operator has two relevant eigenphases ±theta, where
      sin^2(theta/2) = M / N.
- QPE estimates theta/(2*pi) (or 1 - theta/(2*pi), depending on branch).
- From theta we recover
      M = N sin^2(theta/2).
"""

import math
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, Statevector
from qiskit_aer import AerSimulator



# We choose n = 3 search qubits, so N = 2^3 = 8 items.
# We mark exactly two basis states, so M = 2.
#
# Bitstrings are written in the computational basis of the search register.
# You can change these marked states as long as you know M classically.
SEARCH_QUBITS = 3
COUNT_QUBITS = 4

MARKED_STATES = ["011", "101"]  # exactly two solutions
N = 2 ** SEARCH_QUBITS
M_TRUE = len(MARKED_STATES)



# The oracle O acts on the search register by
#     O|x> = -|x>  if x is marked
#     O|x> =  |x>  otherwise.
#
# Since the assignment only asks for a simple Boolean function with known
# number of solutions, we use a diagonal phase oracle. This is mathematically
# clean and easy to verify.
def phase_oracle_unitary(n, marked_states):
    """Return the 2^n x 2^n diagonal oracle matrix."""
    dim = 2 ** n
    diag = np.ones(dim, dtype=complex)

    for x in marked_states:
        idx = int(x, 2)
        diag[idx] = -1.0

    return np.diag(diag)


def oracle_gate(n, marked_states):
    """Construct the oracle as a gate."""
    U = phase_oracle_unitary(n, marked_states)
    return UnitaryGate(U, label="Oracle")


# The Grover diffuser is
#     D = 2|s><s| - I
# where
#     |s> = (1/sqrt(N)) sum_x |x>.
#
# Circuit decomposition:
#   H^n X^n (multi-controlled Z) X^n H^n
#
# We implement the multi-controlled Z through:
#   H on last qubit -> MCX -> H on last qubit
def diffuser_circuit(n):
    """Return the Grover diffuser on n qubits as a QuantumCircuit."""
    qc = QuantumCircuit(n, name="Diffuser")

    qc.h(range(n))
    qc.x(range(n))

    # Implement an n-qubit controlled-Z that flips only |11...1>.
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)

    qc.x(range(n))
    qc.h(range(n))

    return qc


def diffuser_gate(n):
    return diffuser_circuit(n).to_gate(label="Diffuser")


# Grover iteration G = D O
# Convention:
#   First apply oracle O, then diffuser D.
# So the Grover iterate is G = D O.
def grover_iteration_circuit(n, marked_states):
    qc = QuantumCircuit(n, name="Grover")
    qc.append(oracle_gate(n, marked_states), range(n))
    qc.append(diffuser_gate(n), range(n))
    return qc


def grover_gate(n, marked_states):
    return grover_iteration_circuit(n, marked_states).to_gate(label="G")


# Inverse QFT from scratch
# We do not import QFT from the library; instead we implement the textbook
# decomposition.
def inverse_qft_circuit(t):
    """
    Return the inverse QFT on t qubits.

    This version uses:
      - swaps to reverse qubit order,
      - controlled phase rotations,
      - Hadamards.
    """
    qc = QuantumCircuit(t, name="IQFT")

    # Reverse the order of qubits.
    for i in range(t // 2):
        qc.swap(i, t - i - 1)

    # Main inverse-QFT body.
    for j in range(t):
        for m in range(j):
            angle = -np.pi / (2 ** (j - m))
            qc.cp(angle, m, j)
        qc.h(j)

    return qc


def inverse_qft_gate(t):
    return inverse_qft_circuit(t).to_gate(label="IQFT")


# Build the quantum counting circuit
# Layout:
#   counting register: t qubits
#   search register:   n qubits
#
# Steps:
#   (i)  Put counting register into uniform superposition via H^t.
#   (ii) Put search register into |s> via H^n.
#   (iii) Apply controlled powers G^(2^j).
#   (iv) Apply inverse QFT to the counting register.
#   (v)  Measure the counting register.
#
# QPE estimates the eigenphase of G. When the search register overlaps the
# two-dimensional Grover invariant subspace, the measurement concentrates near
# ±theta.
def quantum_counting_circuit(n, t, marked_states):
    counting = QuantumRegister(t, "count")
    search = QuantumRegister(n, "search")
    c_count = ClassicalRegister(t, "c_count")

    qc = QuantumCircuit(counting, search, c_count, name="QuantumCounting")

    # Step (i): counting register to uniform superposition.
    qc.h(counting)

    # Step (ii): search register to uniform superposition |s>.
    qc.h(search)

    # Step (iii): controlled powers of the Grover iterate.
    G = grover_gate(n, marked_states)

    # Standard QPE pattern:
    # qubit j controls G^(2^j). Since Qiskit qubit ordering can be subtle,
    # we keep the convention "count[0] controls G^(1), count[1] controls G^(2), ..."
    for j in range(t):
        power_gate = G.power(2 ** j)
        c_power_gate = power_gate.control(1)
        qc.append(c_power_gate, [counting[j]] + list(search))

    # Step (iv): inverse QFT on counting register.
    qc.append(inverse_qft_gate(t), counting)

    # Step (v): measure only the counting register.
    qc.measure(counting, c_count)

    return qc


# 7. Ideal simulation
def run_ideal_counts(qc, shots=4096):
    """Run on ideal Aer simulator and return counts."""
    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    return result.get_counts()



# 8. Noisy simulation on a FakeBackend 
# The assignment requests a FakeBackend because QPE + IQFT yields a deep circuit.
# Backend class names vary slightly by installed Qiskit version, so we try a
# small list. FakeTorino is attempted first, in line with the project brief.
def get_fake_backend():
    """
    Return an available fake backend.

    Tries FakeTorino first. Falls back to other common fake backends if needed.
    """
    candidates = [
        ("qiskit_ibm_runtime.fake_provider", "FakeTorino"),
        ("qiskit_ibm_runtime.fake_provider", "FakeSherbrooke"),
        ("qiskit_ibm_runtime.fake_provider", "FakeBrisbane"),
        ("qiskit_ibm_runtime.fake_provider", "FakeKyiv"),
        ("qiskit.providers.fake_provider", "FakeManila"),
        ("qiskit.providers.fake_provider", "FakeJakarta"),
        ("qiskit.providers.fake_provider", "FakeLima"),
    ]

    for module_name, class_name in candidates:
        try:
            module = __import__(module_name, fromlist=[class_name])
            backend_class = getattr(module, class_name)
            return backend_class()
        except Exception:
            continue

    raise ImportError(
        "No supported FakeBackend found. Install qiskit-ibm-runtime "
        "or use a version that provides qiskit.providers.fake_provider."
    )


def run_noisy_counts(qc, shots=4096):
    """
    Simulate the circuit on a noisy backend model derived from a FakeBackend.
    """
    fake_backend = get_fake_backend()
    noisy_sim = AerSimulator.from_backend(fake_backend)

    tqc = transpile(qc, noisy_sim, optimization_level=1)
    result = noisy_sim.run(tqc, shots=shots).result()
    return result.get_counts(), fake_backend


# 9. Classical post-processing: phase -> theta -> M
# If the most likely measured integer is y in {0,...,2^t-1}, then the estimated
# phase fraction is
#       phi_hat = y / 2^t.
#
# For Grover, the relevant eigenvalues are e^{±i theta}, so the phase can encode
# either theta/(2*pi) or 1 - theta/(2*pi). We therefore fold the estimate into
# [0, 1/2] by replacing phi with min(phi, 1 - phi).
#
# Then
#       theta_hat = 2*pi*phi_hat_folded
#       M_hat = N sin^2(theta_hat / 2).
def bitstring_to_phase_fraction(bitstring, t):
    """Convert measured bitstring to phase fraction y / 2^t."""
    y = int(bitstring, 2)
    return y / (2 ** t)


def recover_theta_from_phase_fraction(phi):
    """
    Fold phi into [0, 1/2], then return theta = 2*pi*phi_folded.
    """
    phi_folded = min(phi, 1 - phi)
    return 2 * np.pi * phi_folded


def estimate_M_from_bitstring(bitstring, n, t):
    phi = bitstring_to_phase_fraction(bitstring, t)
    theta = recover_theta_from_phase_fraction(phi)
    M_est = (2 ** n) * (np.sin(theta / 2) ** 2)
    return phi, theta, M_est


def most_frequent_outcome(counts):
    return max(counts, key=counts.get)


# 10. Exact theoretical value for comparison
# For the Grover rotation angle theta we have
#       sin^2(theta/2) = M/N.
# Therefore
#       theta = 2 arcsin(sqrt(M/N)).
def exact_theta_from_M(M, N):
    return 2 * np.arcsin(np.sqrt(M / N))


def exact_phase_fraction_from_M(M, N):
    theta = exact_theta_from_M(M, N)
    return theta / (2 * np.pi)


# 11. Plotting utilities
def counts_to_probability_vector(counts, t, shots):
    vec = np.zeros(2 ** t)
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        vec[idx] = count / shots
    return vec


def plot_distributions(ideal_counts, noisy_counts, t, shots, filename="quantum_counting_distributions.png"):
    x = np.arange(2 ** t)
    ideal_probs = counts_to_probability_vector(ideal_counts, t, shots)
    noisy_probs = counts_to_probability_vector(noisy_counts, t, shots)

    plt.figure(figsize=(12, 5))
    width = 0.42

    plt.bar(x - width / 2, ideal_probs, width=width, label="Ideal")
    plt.bar(x + width / 2, noisy_probs, width=width, label="Noisy (FakeBackend)")

    plt.xlabel("Measured integer y")
    plt.ylabel("Probability")
    plt.title("Quantum Counting: ideal vs noisy phase distributions")
    plt.xticks(x, [format(i, f"0{t}b") for i in x], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


# 12. State-space sanity check
# This section is not necessary for execution, but it is useful to verify the
# eigenvalue structure numerically for the project report.
def grover_eigen_analysis(n, marked_states):
    """
    Numerically compute the eigenvalues of the Grover operator and return them.
    Useful for confirming the ±theta phase structure.
    """
    G = Operator(grover_gate(n, marked_states)).data
    eigvals, _ = np.linalg.eig(G)
    return eigvals



# 13. Main routine
def main():
    shots = 4096

    print("=" * 70)
    print("Quantum Counting project")
    print("=" * 70)
    print(f"Search qubits n          : {SEARCH_QUBITS}")
    print(f"Counting qubits t        : {COUNT_QUBITS}")
    print(f"N = 2^n                  : {N}")
    print(f"Marked states            : {MARKED_STATES}")
    print(f"True number of solutions : {M_TRUE}")
    print()

    theta_exact = exact_theta_from_M(M_TRUE, N)
    phi_exact = exact_phase_fraction_from_M(M_TRUE, N)

    print(f"Exact theta = 2*arcsin(sqrt(M/N)) = {theta_exact:.8f} rad")
    print(f"Exact phase fraction theta/(2*pi) = {phi_exact:.8f}")
    print()

    qc = quantum_counting_circuit(SEARCH_QUBITS, COUNT_QUBITS, MARKED_STATES)
    print("Quantum counting circuit:")
    print(qc.draw(output="text"))
    print()

    # Ideal run
    ideal_counts = run_ideal_counts(qc, shots=shots)
    ideal_best = most_frequent_outcome(ideal_counts)
    ideal_phi, ideal_theta, ideal_M = estimate_M_from_bitstring(
        ideal_best, SEARCH_QUBITS, COUNT_QUBITS
    )

    print("Ideal simulator results:")
    print(ideal_counts)
    print(f"Most frequent outcome       : {ideal_best}")
    print(f"Recovered phase fraction    : {ideal_phi:.8f}")
    print(f"Recovered theta             : {ideal_theta:.8f} rad")
    print(f"Recovered M                 : {ideal_M:.8f}")
    print(f"Rounded recovered M         : {round(ideal_M)}")
    print()

    # Noisy run
    noisy_counts, fake_backend = run_noisy_counts(qc, shots=shots)
    noisy_best = most_frequent_outcome(noisy_counts)
    noisy_phi, noisy_theta, noisy_M = estimate_M_from_bitstring(
        noisy_best, SEARCH_QUBITS, COUNT_QUBITS
    )

    print(f"Noisy simulation backend    : {fake_backend.name}")
    print("Noisy simulator results:")
    print(noisy_counts)
    print(f"Most frequent outcome       : {noisy_best}")
    print(f"Recovered phase fraction    : {noisy_phi:.8f}")
    print(f"Recovered theta             : {noisy_theta:.8f} rad")
    print(f"Recovered M                 : {noisy_M:.8f}")
    print(f"Rounded recovered M         : {round(noisy_M)}")
    print()

    # Save the comparison plot.
    plot_distributions(
        ideal_counts,
        noisy_counts,
        COUNT_QUBITS,
        shots,
        filename="quantum_counting_distributions.png",
    )
    print("Saved plot: quantum_counting_distributions.png")
    print()

    # Optional eigen-analysis.
    eigvals = grover_eigen_analysis(SEARCH_QUBITS, MARKED_STATES)
    print("Numerical eigenvalues of Grover operator G:")
    print(eigvals)
    print()

    print("Done.")


if __name__ == "__main__":
    main()
