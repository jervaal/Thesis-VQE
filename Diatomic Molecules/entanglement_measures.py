import numpy as np

from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, entanglement_of_formation
from qiskit.circuit import QuantumCircuit

import itertools


#@jit
def Bipartite_Negativity(rho ):
    
    matrix = rho.data
    #print(matrix)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    return sum([ ( np.abs(eigenvalue) - eigenvalue )/2 for eigenvalue in eigenvalues ])
    
    
#@jit
def Multipartite_Negativity(circuit : QuantumCircuit, mode='max' ):

    #possible combinations: N choose 2
    #because choosing ij = ji ---> 
    #calculate [(N Choose 2) / 2] possible outcomes
    
    qubits_list = circuit.qubits
    rho = DensityMatrix(circuit)
    
    index_list = [qubit._index for qubit in qubits_list]

    num_q = len(qubits_list)

    negativity_dict = {}

    if num_q == 2:
        return Bipartite_Negativity(rho)

    else:
        qubits_list = circuit.qubits
    
        index_list = np.array([qubit._index for qubit in qubits_list])

        negativity_matrix = np.zeros(shape= (num_q, num_q))

        for qbit_indx1, ind_i in enumerate(index_list):
            
            for qbit_indx2, ind_j in enumerate(index_list[:ind_i]):
                
                qubit_trace_list =  [k for k in index_list if k not in (qbit_indx1, qbit_indx2)]
                rho_p = partial_trace(rho, qubit_trace_list)

                rho_p_Ta = rho_p.partial_transpose([0, 1])

                negativity_matrix[ind_i][ind_j] = Bipartite_Negativity(rho_p)
    
    if mode== 'max':
        return negativity_matrix.max()
    
    elif mode== 'total':
        return np.sum(negativity_matrix)


#@jit
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

    #por implementar: 
    # 1) Caso especial reduccion de 2 qubits
    # calcular los coefs de Schmidt y compararlos

    # 2) Hacer la equivalencia unitaria entre los dos estados
    # (Computacionalmente no se me ocurre de que manera se podria hacer) sin que sea 
    # Costoso computacionalmente.


    return np.square(np.abs(GHZ_state.inner(state)))


#@jit
def multipartite_entanglement(circuit : QuantumCircuit,  rho : DensityMatrix, mode= 'max'):
    
    if not rho:
        rho = DensityMatrix(circuit)

    dimention = rho.dim
    if dimention == 4:
        return entanglement_of_formation(rho)
    
    
    else:
        qubits_list = circuit.qubits
    
        index_list = np.array([qubit._index for qubit in qubits_list])

        num_q = int(np.sqrt(dimention))
        entglmnt_matrix = np.zeros(shape= (num_q, num_q))

        for qbit_indx1, ind_i in enumerate(index_list):
            
            for qbit_indx2, ind_j in enumerate(index_list[:ind_i]):
                
                qubit_trace_list =  [k for k in index_list if k not in (qbit_indx1, qbit_indx2)]
                rho_p = partial_trace(rho, qubit_trace_list)

                entglmnt_matrix[ind_i][ind_j] = entanglement_of_formation(rho_p)
    
    if mode== 'max':
        return entglmnt_matrix.max()
    
    elif mode== 'total':
        return np.sum(entglmnt_matrix)

