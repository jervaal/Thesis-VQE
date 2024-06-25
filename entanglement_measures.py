import numpy as np


from qiskit.quantum_info import DensityMatrix, partial_trace, Statevector
from qiskit.circuit import QuantumCircuit

import itertools



def Bipartite_Negativity(rho : DensityMatrix):
    
    matrix = rho.data

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    return sum([ ( np.abs(eigenvalue) - eigenvalue )/2 for eigenvalue in eigenvalues ])
    
    

def Multipartite_Negativity(qcircuit : QuantumCircuit, mode='max' ):

    #possible combinations: N choose 2
    #because choosing ij = ji ---> 
    #calculate [(N Choose 2) / 2] possible outcomes
    
    qubits_list = qcircuit.qubits
    rho = DensityMatrix(qcircuit)
    
    index_list = [qubit._index for qubit in qubits_list]


    negativity_dict = {}


    for i in index_list:
        ind_i = index_list.index(i)
        for j in index_list[:ind_i]:
            qubit_trace_list =  [k for k in index_list if k not in (i, j)]

            rho_p = partial_trace(rho, qubit_trace_list)

            negativity_dict[f'{i}{j}'] = Bipartite_Negativity(rho_p)
    
    if mode== 'max':
        return max(negativity_dict.values())
    
    elif mode== 'total':
        return sum(negativity_dict.values())



def GHZ_entanglmenetlike_measure(qcircuit : QuantumCircuit):

    #state
    num_qubits = qcircuit.num_qubits

    state = Statevector(qcircuit)

    # generate_GHZ:

    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    for i, j in zip(range(num_qubits-1), range(1,num_qubits)):
        qc.cx(i, j)
    
    GHZ_state = Statevector(qc)

    return np.square(np.abs(GHZ_state.inner(state)))


