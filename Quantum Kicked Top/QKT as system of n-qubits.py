import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm,expm,logm,fractional_matrix_power

sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
I2=np.eye(2)

def get_Jz(n): # n: number of qubits
  Jz = np.zeros((2**n, 2**n), dtype=complex)
  for i in range(n):
    # Create a list of identity matrices
    operators = [np.eye(2) for _ in range(n)]
    # Replace the i-th operator with sigma_z
    operators[i] = sigma_z
    # Calculate the tensor product of all operators
    temp_operator = operators[0]
    for j in range(1, n):
      temp_operator = np.kron(temp_operator, operators[j])
    # Add the current operator to Jz
    Jz += temp_operator
  return Jz/2

def get_Jx(n):
  Jx = np.zeros((2**n, 2**n), dtype=complex)
  for i in range(n):
    # Create a list of identity matrices
    operators = [np.eye(2) for _ in range(n)]
    # Replace the i-th operator with sigma_z
    operators[i] = sigma_x
    # Calculate the tensor product of all operators
    temp_operator = operators[0]
    for j in range(1, n):
      temp_operator = np.kron(temp_operator, operators[j])
    # Add the current operator to Jz
    Jx += temp_operator
  return Jx/2

def get_Jy(n):
  Jy = np.zeros((2**n, 2**n), dtype=complex)
  for i in range(n):
    # Create a list of identity matrices
    operators = [np.eye(2) for _ in range(n)]
    # Replace the i-th operator with sigma_z
    operators[i] = sigma_y
    # Calculate the tensor product of all operators
    temp_operator = operators[0]
    for j in range(1, n):
      temp_operator = np.kron(temp_operator, operators[j])
    # Add the current operator to Jz
    Jy += temp_operator
  return Jy/2

pnoise_level = 0

def get_U(p_val, k_val, nq): 
  operator1 = get_Jy(nq)
  operator2 = np.matmul(get_Jz(nq), get_Jz(nq))
  A = expm(-1j * p_val * operator1 * (1+np.random.normal(0, pnoise_level)))
  B = expm(-1j * (k_val / nq) * operator2)  # nq (number of qubits) = 2j (QKT spin)
  U = np.dot(B, A)
  return U

def generate_state(theta, phi, nq):
  # Initialize state in |00000>
  state = np.zeros(2**nq, dtype=np.complex128)
  state[0] = 1

  # Rotation around y-axis
  Ry = expm(-1j * theta/2 * sigma_y * (1+np.random.normal(0, pnoise_level)))

  # Rotation around z-axis
  Rz =  expm(-1j * np.pi/4 * sigma_x *(1+np.random.normal(0, pnoise_level))) @ expm(-1j * phi/2 * sigma_y * (1+np.random.normal(0, pnoise_level))) @ expm(1j * np.pi/4 * sigma_x * (1+np.random.normal(0, pnoise_level)))


  # Apply rotations to each qubit
  gate = Ry
  for _ in range(nq - 1):
    gate = np.kron(gate, Ry)
  state = np.dot(gate, state)

  gate = Rz
  for _ in range(nq - 1):
    gate = np.kron(gate, Rz)
  state = np.dot(gate, state)
  return state

def evolve(theta,phi,time, p, k, nq):
  psi = generate_state(theta, phi, nq)
  Jx=[]
  #Jy=[]
  op1 = get_Jx(nq)
  for t in range(time):
    U = get_U(p,k,nq)
    result = np.matmul(psi.conjugate().transpose(), np.matmul(op1/(nq/2) , psi))
    psi = np.matmul(U ,psi)
    Jx.append(np.real(result))
  return np.array(Jx)

def evolve_average(theta, phi, time, p, k, nq, num_averages):
  results = []
  for _ in range(num_averages):
    results.append(evolve(theta, phi, time, p, k, nq))
  average_results = np.mean(results, axis=0)
  return average_results

runtime=100
fig = plt.figure(figsize=(12, 3))
plt.plot(np.arange(runtime),evolve_average(2.25,0.63,100,np.pi/2,10,3,1),markersize=3, linewidth=0.5)
