%pip install qiskit
!pip install qiskit-aer

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector

from qiskit.primitives import Estimator
from qiskit_aer import Aer
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qiskit.compiler import transpile
import time

    #j=3/2 #alpha=k/3
    N_H = 100
    theta = 2.25
    phi = 0.63
    alpha = 10/3

    qreg = QuantumRegister(3, 'q')

    # Sub-circuit as Hamiltonian implementation
    kick = QuantumCircuit(qreg, name = 'kick')
    kick.ry(np.pi/2,qreg)

    kick.cx(qreg[1],qreg[0])
    kick.rz(alpha,qreg[0])
    kick.cx(qreg[1],qreg[0])

    kick.cx(qreg[2],qreg[1])
    kick.rz(alpha,qreg[1])
    kick.cx(qreg[2],qreg[1])

    kick.cx(qreg[2],qreg[0])
    kick.rz(alpha,qreg[0])
    kick.cx(qreg[2],qreg[0])

    Jx = np.zeros(N_H)

    qc = QuantumCircuit(qreg)
      # rotating qubits to starting position
    qc.ry(theta, qreg)
    qc.rz(phi, qreg)
    pauli_terms = ["XII", "IXI", "IIX"]
    coeffs = [1/2, 1/2, 1/2]
    op = SparsePauliOp(pauli_terms, coeffs)

    for j in range(N_H):
      qc.append(kick, qreg)
      statevector = Statevector(qc)
      psi=np.array(statevector)
      Jx[j] = (np.matmul(psi.conjugate().transpose(), np.matmul(op.to_matrix()/1.5 , psi))).real
    fig = plt.figure(figsize=(12, 3))
    ax1 = fig.add_subplot(111)
    timesteps = [*range(N_H)]
    ax1.plot(timesteps, Jx,markersize=3, linewidth=0.5)

kick.draw()
