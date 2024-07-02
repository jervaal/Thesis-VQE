# General imports
import numpy as np

# Pre-defined ansatz circuit and operator class for Hamiltonian
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp

# SciPy minimizer routine
from scipy.optimize import minimize

# Plotting functions
import matplotlib.pyplot as plt

from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import EstimatorV2 as Estimator


from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime.fake_provider import FakeManilaV2




def cost_func(params, ansatz, hamiltonian, estimator, cost_history_dict):

    """Return cost function for optimization

    Parameters:
        params (ndarray): Array of ansatz parameters
        list_coefficients (list): List of arrays of complex coefficients
        list_labels (list): List of labels
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        obs (SparsePauliOp): Observable
        estimator (EstimatorV2): Statevector estimator primitive instance
        pm (PassManager): Pass manager
        callback_dict (dict): Dictionary to store callback information

    Returns:
        float: Cost function estimate
    """

    #ansatz + initial state ready to run over a Hamiltonian and being optimized


    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]



    cost_history_dict['iters'] += 1
    cost_history_dict['prev_vector'] = params
    cost_history_dict['cost_value'] = 0
    
    return energy


def Initialize_VQE_hardware_EfficientSU2(sparse_pauli_hamiltonian, backend_name= 'ibm_brisbane'):
    
    service = QiskitRuntimeService()
    

    ansatz = EfficientSU2(sparse_pauli_hamiltonian.num_qubits,
                      #su2_gates= ['rz'], #architecture of ibm_brisbane
                      entanglement= 'linear',
                       reps= 4 #this is for the parity mapper
                      ) #try what happens if cz

    backend= FakeManilaV2()

    target = backend.target
    pm = generate_preset_pass_manager(target= target, optimization_level= 3)

    ansatz_isa = pm.run(ansatz)

    hamiltonian_isa = sparse_pauli_hamiltonian.apply_layout(layout= ansatz_isa.layout)

    initial_point = np.random.random(ansatz.num_parameters)

    with Session(backend= backend) as session:
        estimator= Estimator(session= session)
        estimator.options.default_shots= 1024

        cost_history_dict = {'iters' : 0,
                         'prev_vector' : None,
                         'cost_value' : 0}

        res = minimize(
            cost_func,
            initial_point,
            args=(ansatz_isa, hamiltonian_isa, estimator, cost_history_dict),
            method= 'cobyla'
        )
    
    return res

#lets try it:

h = SparsePauliOp.from_list(
    [("YZ", 0.3980), ("ZI", -0.3980), ("ZZ", -0.0113), ("XX", 0.1810)]
)

results = Initialize_VQE_hardware_EfficientSU2(h)

print(results)
