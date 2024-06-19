
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import InstructionProperties
from qiskit.visualization import plot_distribution
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.primitives import StatevectorEstimator

from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import (
    QiskitRuntimeService, 
    EstimatorV2 as Estimator,
    SamplerV2 as Sampler,
    EstimatorOptions
)


def cost_func(params, initial_state_circuit, ansatz, obs, estimator, pm, callback_dict, qn= False):

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


    qc = initial_state_circuit
    mesurador = qc.compose(ansatz)
    
    if qn:
        mesurador = ansatz
        #with qiskit nature the UCCSD type of anstaz
        #is already done
    
    mesurador.draw('mpl')

    transpiled_mesurador = pm.run(mesurador)
    transpiled_hamiltonian = obs.apply_layout(layout= transpiled_mesurador.layout)

    pub = (transpiled_mesurador, transpiled_hamiltonian, params)

    job = estimator.run([pub])



    result = job.result()[0]
    energy = result.data.evs



    callback_dict["iters"] += 1
    callback_dict["prev_vector"] = params
    callback_dict["cost_history"].append(energy)

    # Print the iterations to screen on a single line
    #print(
    #    "Iters. done: {} [Current energy {}]".format(callback_dict["iters"], energy),
     #   end="\r",
    #    flush=True,
    #)
    
    return energy


