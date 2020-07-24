import numpy as np
from qiskit.circuit import Gate, QuantumRegister
from qiskit.circuit.library.standard_gates import CU1Gate, CXGate, HGate, MCXGate, RYGate, RZGate, SwapGate, XGate, ZGate


def get_num_qubits(N):
    return int(np.log2(N - 0.5) + 1.0)

class CustomMCXGate(Gate):
    """A MCX gate with a tunable number of ancilla qubits."""

    def __init__(self, num_ctrl_qubits, qubit_values=None, label=None):
        self.num_ctrl_qubits = num_ctrl_qubits
        self.qubit_values = qubit_values

        super().__init__(name="CustomMCX", num_qubits=self.num_ctrl_qubits + 1, label=label, params=[])

    def _define(self):
        definition = []
        qr = QuantumRegister(self.num_ctrl_qubits + 1)

        ctrl_qr = qr[:self.num_ctrl_qubits]
        target_qubit = qr[self.num_ctrl_qubits]

        if self.qubit_values:
            for qubit_index, qubit_value in enumerate(self.qubit_values):
                if not qubit_value:
                    definition.append((XGate(), [ctrl_qr[qubit_index]], []))

        definition.append((MCXGate(self.num_ctrl_qubits), list(ctrl_qr) + [target_qubit], []))

        if self.qubit_values:
            for qubit_index, qubit_value in enumerate(self.qubit_values):
                if not qubit_value:
                    definition.append((XGate(), [ctrl_qr[qubit_index]], []))

        self.definition = definition

class BooleanOracleGate(Gate):
    """Implements a quantum gate that represents a boolean map."""

    def __init__(self, boolean_vector, label=None):
        self.num_ctrl_qubits = get_num_qubits(len(boolean_vector))

        if len(boolean_vector) != 2**self.num_ctrl_qubits:
            raise ValueError("Mismatch between size of given vector and expected vector size given number of qubits.")

        self.boolean_vector = boolean_vector

        super().__init__("BooleanOracleGate", self.num_ctrl_qubits + 1, label=label, params=[])

    def _define(self):
        definition = []
        qr = QuantumRegister(self.num_ctrl_qubits + 1)

        for (idx, val) in enumerate(self.boolean_vector):
            if val:
                qubit_values = ([int(bit) for bit in f"{{:0{self.num_ctrl_qubits}b}}".format(idx)[::-1]] if self.num_ctrl_qubits else [])
                gate = CustomMCXGate(self.num_ctrl_qubits, qubit_values)
                definition.append((gate, qr, []))

        self.definition = definition

class QFTGate(Gate):
    """A gate implementing the standard Quantum Fourier Transform (QFT)."""

    def __init__(self, num_qubits, label=None, do_swaps=True):
        super().__init__(name=f"QFTGate_{num_qubits}", num_qubits=num_qubits, label=label, params=[])
        self.do_swaps = do_swaps

    def _define(self):
        definition = []
        qr = QuantumRegister(self.num_qubits)

        for i in reversed(range(self.num_qubits)):
            definition.append((HGate(), [qr[i]], []))
            for j in range(i):
                definition.append((CU1Gate(np.pi / 2.0**(i - j)), [qr[j], qr[i]], []))

        if self.do_swaps:
            for i in range(self.num_qubits // 2):
                definition.append((SwapGate(), [qr[i], qr[self.num_qubits - 1 - i]], []))

        self.definition = definition

class GroverDiffusionGate(Gate):

    def __init__(self, num_qubits, quantum_algorithm, classical_boolean_oracle, oracle_qubits, minus_identity=False, label=None):
        super().__init__("GroverDiffusionGate", num_qubits + 1, label=label, params=[])
        self.quantum_algorithm = quantum_algorithm
        self.classical_boolean_oracle = classical_boolean_oracle
        self.oracle_qubits = oracle_qubits
        self.minus_identity = minus_identity

    def _define(self):
        definition = []
        qr = QuantumRegister(self.num_qubits)
        oracle_qr = [qr[i] for i in self.oracle_qubits]

        # Prepare the auxiliary qubit into the Hadamard minus state.
        definition.append((HGate(), [qr[-1]], []))
        definition.append((ZGate(), [qr[-1]], []))

        # Add the boolean oracle whose behavior is
        # determined by the auxiliary qubit.
        definition.append((BooleanOracleGate(self.classical_boolean_oracle), oracle_qr + [qr[-1]], []))

        # Add the inverse of the quantum algorithm.
        definition.append((self.quantum_algorithm.inverse(), qr[:-1], []))

        # Add the 'inversion around the mean' operator as a
        # special boolean oracle that behaves as a phase oracle
        # because of how was set the auxiliary qubit.
        definition.append((BooleanOracleGate([1] + [0] * (2 ** (self.num_qubits - 1) - 1)), qr[:], []))

        # Add the quantum algorithm.
        definition.append((self.quantum_algorithm, qr[:-1], []))

        # Revert the auxiliary qubit to zero.
        definition.append((ZGate(), [qr[-1]], []))
        definition.append((HGate(), [qr[-1]], []))

        self.definition = definition

class QPEGate(Gate):
    """A gate implementing the standard Quantum Phase Estimation (QPE)."""

    def __init__(self, num_phase_qubits, unitary_gate, label=None):
        super().__init__(f"QPE_{num_phase_qubits}", num_phase_qubits + unitary_gate.num_qubits, params=[])
        self.num_phase_qubits = num_phase_qubits
        self.unitary_gate = unitary_gate

    def _define(self):
        definition = []
        # qr[:self.num_phase_qubits]: phase qubits
        # qr[self.num_phase_qubits:]: state qubits
        qr = QuantumRegister(self.num_qubits)

        # Loop over the phase qubits.
        for i in range(self.num_phase_qubits):
            definition.append((HGate(), [qr[i]], []))
            for _ in range(2**i):
                definition.append((self.unitary_gate.control(num_ctrl_qubits=1), [qr[i]] + qr[self.num_phase_qubits:], []))

        definition.append((QFTGate(self.num_phase_qubits).inverse(), qr[:self.num_phase_qubits], []))

        self.definition = definition
