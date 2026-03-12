import dimod
from braket.circuits import Observable
from braket.registers.qubit import Qubit
from braket.registers.qubit_set import QubitSet
import numpy as np
from scipy.optimize import minimize
from braket.circuits import Circuit
from braket.devices import LocalSimulator

def observable_to_str(observable, variables):
	if isinstance(observable, Observable.Sum):
		s = [observable_to_str(x, variables) for x in observable.summands]
		return " + ".join(s)
	elif isinstance(observable, Observable.TensorProduct):
		c = observable.coefficient
		s = [observable_to_str(x, variables) for x in observable.factors]
		return (str(c) + " " if c != 1.0 else "") + " x ".join(s)
	elif isinstance(observable, Observable.Z):
		c = observable.coefficient
		return (str(c) + " " if c != 1.0 else "") + "Z(" + qubits_to_str(observable.targets, variables) + ")"
	elif isinstance(observable, Observable.Y):
		c = observable.coefficient
		return (str(c) + " " if c != 1.0 else "") + "Y(" + qubits_to_str(observable.targets, variables) + ")"
	elif isinstance(observable, Observable.X):
		c = obvserable.coefficient
		return (str(c) + " " if c != 1.0 else "") + "X(" + qubits_to_str(observable.targets, variables) + ")"
	else:
		c = observable.coefficient
		return (str(c) + " " if c != 1.0 else "") + str(observable)

def qubits_to_str(qubits, variables):
	if isinstance(qubits, QubitSet):
		if len(qubits) == 0:
			return ""
		s = [qubits_to_str(x, variables) for x in qubits]
		return ", ".join(s)
	elif isinstance(qubits, Qubit):
		return str(variables[int(qubits)])
	else:
		return None

def bqm_to_braket_hamiltonian(bqm: dimod.BinaryQuadraticModel):
    """
    Convert a D-Wave BQM into a Hamiltonian usable as the QAOA cost
    observable in Amazon Braket.

    Returns
    -------
    observable : Observable
        Braket observable representing the cost Hamiltonian
    offset : float
        Constant energy offset from the BQM
    variable_order : list
        Mapping from qubit index -> BQM variable
    """

    # Convert BQM to Ising form
    h, J, offset = bqm.to_ising()

    # Fix variable ordering
    variables = bqm.variables
    index = {v: i for i, v in enumerate(variables)}

    observable = None

    # Linear terms
    for v, bias in h.items():
        if abs(bias) == 0:
            continue

        term = bias * Observable.Z(index[v])

        observable = term if observable is None else observable + term

    # Quadratic terms
    for (u, v), bias in J.items():
        if abs(bias) == 0:
            continue

        term = bias * (Observable.Z(index[u]) @ Observable.Z(index[v]))

        observable = term if observable is None else observable + term

    return observable, offset, variables
    
def flatten(obs, coeff=1.0):
    """Recursively flatten the Hamiltonian into (coeff, base_observable) pairs."""
    if hasattr(obs, "summands"):
        # Sum of observables
        for s in obs.summands:
            yield from flatten(s, coeff)
    else:
        yield coeff * obs.coefficient, obs

def apply_hamiltonian_exponent(circ: Circuit, hamiltonian, gamma: float):
    """
    Apply e^{-i gamma H} for a Braket Hamiltonian to a circuit.

    Parameters:
    - circ: Circuit to append gates to
    - hamiltonian: Braket observable (Z, X, Y, TensorProduct, Sum)
    - gamma: evolution parameter
    """
    
    for coeff, obs in flatten(hamiltonian):
        angle = 2 * gamma * coeff  # factor of 2 for Braket RZ convention

        # Determine qubits and Pauli types
        if obs.name == "TensorProduct":
            factors = obs.factors
        else:
            factors = [obs]

        paulis = [f.name for f in factors]
        qubits = [f.targets[0] for f in factors]  # treat qubit as integer

        # Basis change: convert all Paulis to Z
        for p, q in zip(paulis, qubits):
            if p == "X":
                circ.h(q)
            elif p == "Y":
                circ.rx(q, -np.pi / 2)

        # Entangle chain
        for i in range(len(qubits) - 1):
            circ.cnot(qubits[i], qubits[i + 1])

        # Apply RZ rotation on last qubit
        circ.rz(qubits[-1], angle)

        # Uncompute entanglement
        for i in reversed(range(len(qubits) - 1)):
            circ.cnot(qubits[i], qubits[i + 1])

        # Undo basis change
        for p, q in zip(paulis, qubits):
            if p == "X":
                circ.h(q)
            elif p == "Y":
                circ.rx(q, np.pi / 2)

def measure_hamiltonian_expectation(circ, hamiltonian, device, shots=1000):
    """
    Compute the expectation value of a Hamiltonian as a sum of weighted observables
    using an existing circuit.

    Parameters:
    - circ: Braket Circuit, already prepared with your QAOA ansatz or state
    - hamiltonian: list of (coefficient, observable) pairs
    - shots: number of samples for each observable

    Returns:
    - estimated expectation value of the full Hamiltonian
    """
    
    
    
    total_expectation = 0.0

    for coeff, obs in flatten(hamiltonian):
        # Copy the existing circuit to avoid modifying the original
        circ_copy = circ.copy()

        # Determine qubits involved in this observable
        if obs.name == "TensorProduct":
            qubits = [f.targets[0] for f in obs.factors]
        else:
            qubits = [obs.targets[0]]

        # Apply basis change for X or Y measurements
        if obs.name == "X":
            for q in qubits:
                circ_copy.h(q)
        elif obs.name == "Y":
            for q in qubits:
                circ_copy.rx(q, -np.pi/2)
        elif obs.name == "TensorProduct":
            for f in obs.factors:
                q = f.targets[0]
                if f.name == "X":
                    circ_copy.h(q)
                elif f.name == "Y":
                    circ_copy.rx(q, -np.pi/2)
                # Z requires no change

        # Measure all qubits of this observable
        for q in qubits:
            circ_copy.measure(q)

        # Run the circuit
        task = device.run(circ_copy, shots=shots)
        result = task.result()
        counts = result.measurement_counts

        # Compute expectation from sampled counts
        exp = 0.0
        for bitstring, count in counts.items():
            # Map '0' -> +1, '1' -> -1
            bit_vals = [1 if b == '0' else -1 for b in bitstring[::-1]]
            obs_val = 1
            for val in bit_vals[:len(qubits)]:
                obs_val *= val
            exp += obs_val * count
        exp /= shots

        # Add weighted contribution
        total_expectation += coeff * exp

    return total_expectation

def build_qaoa_circuit(params, num_qubits, hamiltonian, p):
    gammas = params[:p]
    betas = params[p:]

    circ = Circuit()
    for q in range(num_qubits):
        circ.h(q)

    for layer in range(p):
        apply_hamiltonian_exponent(circ, hamiltonian, gammas[layer])
        for q in range(num_qubits):
            circ.rx(q, 2 * betas[layer])
    
    return circ
    
def qaoa_expectation(params, num_qubits, hamiltonian, p, device):
    circ = build_qaoa_circuit(params, num_qubits, hamiltonian, p)

    return measure_hamiltonian_expectation(circ, hamiltonian, device, shots=1000)

def run_qaoa(hamiltonian, num_qubits, p=1):

    device = LocalSimulator(backend="braket_sv")

    params0 = np.random.uniform(0, np.pi, 2 * p)

    res = minimize(
        qaoa_expectation,
        params0,
        args=(num_qubits, hamiltonian, p, device),
        method="COBYLA",
    )

    return res
    
def evaluate_bitstring(bitstring, hamiltonian):
    value = 0
    for coeff, obs in flatten(hamiltonian):
        if obs.name == "TensorProduct":
            qubits = [f.targets[0] for f in obs.factors]
        else:
            qubits = [obs.targets[0]]
        obs_val = 1
        for q in qubits:
            obs_val *= 1 if bitstring[::-1][q] == '0' else -1
        value += coeff * obs_val
    return value
    
def qaoa(hamiltonian, num_qubits, p = 1):
	result = run_qaoa(hamiltonian, num_qubits, p=p)

	print("Optimal parameters:", result.x)
	print("Minimum expectation:", result.fun)

	circ_opt = build_qaoa_circuit(result.x, num_qubits, hamiltonian, p)

	device = LocalSimulator()
	task = device.run(circ_opt, shots=5000)  # increase shots for better statistics
	result = task.result()
	counts = result.measurement_counts

	optimal_bitstring = min(counts.keys(), key=lambda b: evaluate_bitstring(b, hamiltonian))
	optimal_energy = evaluate_bitstring(optimal_bitstring, hamiltonian)
	
	return optimal_bitstring, optimal_energy
	
	
	
