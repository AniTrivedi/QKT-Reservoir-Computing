import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm,expm,logm,fractional_matrix_power
!pip install qutip
from qutip import partial_transpose, Qobj

sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
I2=np.eye(2)

def get_Jz(n):
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
  B = expm(-1j * (k_val / nq) * operator2)
  U = np.dot(B, A)
  return U



def Ry(theta):
  Ry = expm(-1j * theta/2 * sigma_y * (1+np.random.normal(0, pnoise_level)))
  return Ry

  # Rotation around z-axis
def Rz(phi):
  Rz =  expm(-1j * np.pi/4 * sigma_x *(1+np.random.normal(0, pnoise_level))) @ expm(-1j * phi/2 * sigma_y * (1+np.random.normal(0, pnoise_level))) @ expm(1j * np.pi/4 * sigma_x * (1+np.random.normal(0, pnoise_level)))
  return Rz


def logarithmic_negativity(rho):
    # Convert Qobj to density matrix
    rho_dm = rho
    rho_PT = partial_transpose(rho_dm, [0, 1]).full()
    eigenvalues = np.linalg.eigvals(rho_PT)
    negativity = np.abs(np.sum(eigenvalues[eigenvalues < 0]))
    log_neg = np.log2(2 * negativity + 1)
    return log_neg

# Define Pauli matrices
sigmax = np.array([[0, 1], [1, 0]])
sigmay = np.array([[0, -1j], [1j, 0]])
sigmaz = np.array([[1, 0], [0, -1]])
identity = np.array([[1, 0], [0, 1]])

# Tensor product for 2-qubit systems
def tensor(A, B):
    return np.kron(A, B)

#Paramaterized Unitary Uc for state generation
num_states = 100
thetas = np.linspace(0, np.pi, num_states)
alphas = np.linspace(0, 2 * np.pi, num_states)

# Data storage
test_states = [[0 for i in range(num_states)] for j in range(num_states)]
prep_ops = [[0 for i in range(num_states)] for j in range(num_states)]  #Uc
ent_labels = [[0 for i in range(num_states)] for j in range(num_states)]
log_neg = [[0 for i in range(num_states)] for j in range(num_states)]

# Constants
omega = 1
beta = 1.3
J = 8.7

# Single-qubit Hamiltonian for thermal state
H1 = -0.5 * omega * sigmaz
thermal_state = expm(-beta * H1)
thermal_state = np.kron(thermal_state, thermal_state)
thermal_state = thermal_state / np.trace(thermal_state)

# Two-qubit Hamiltonian for interaction
H = 2 * np.pi * J * tensor(sigmaz/2, sigmaz/2)

# Main loop over thetas and alphas
for j in range(len(alphas)):
    for i in range(len(thetas)):

        # First qubit rotation around Y-axis by theta
        p1 = expm(-1j * thetas[i] * np.kron(sigmay/2, identity))

        p2 = expm(-1j * np.pi/2 * np.kron(identity,sigmax/2))

        # Interaction evolution
        cnot_evolve = expm(-1j * H * (1 / (2 * J) / (np.pi/2)) * alphas[j])

        # Rotation around Y-axis of second qubit (Pauli-Y)
        p3 = expm(-1j * np.pi/2 * np.kron(identity, sigmay/2))

        # Combine operations to form the total unitary
        Uc = np.dot(p3, np.dot(cnot_evolve, np.dot(p2, p1)))

        # Evolve the thermal state
        new_state = Qobj(Uc @ thermal_state @ np.conjugate(Uc.T),dims=[[2, 2], [2, 2]])

        # Calculate logarithmic negativity
        log_neg[j][i] = logarithmic_negativity(new_state)

        # Label as entangled or not (1 for entangled, 0 for not)
        label = 1 if log_neg[j][i] > 0 else 0

        new_state = new_state.full()

        # Store results
        test_states[j][i] = new_state
        prep_ops[j][i] = Uc
        ent_labels[j][i] = label

# Actual states possible to generate in NMR lab
num_states = 100
thetas = np.linspace(0, np.pi, num_states)
alphas = np.linspace(0, 2 * np.pi, num_states)

# Data storage
test_states = [[0 for i in range(num_states)] for j in range(num_states)]

# Constants
omega = 500*10**6
beta = 1.6/10**(13)
J = 8.7

# Single-qubit Hamiltonian for thermal state
H1 = -0.5 * omega * sigmaz
thermal_state = expm(-beta * H1)
thermal_state = np.kron(thermal_state, thermal_state)
thermal_state = thermal_state / np.trace(thermal_state)

# Two-qubit Hamiltonian for interaction
H = 2 * np.pi * J * tensor(sigmaz/2, sigmaz/2)

# Main loop over thetas and alphas
for j in range(len(alphas)):
    for i in range(len(thetas)):
        Uc = prep_ops[j][i]

        # Evolve the thermal state
        new_state = Qobj(Uc @ thermal_state @ np.conjugate(Uc.T),dims=[[2, 2], [2, 2]])

        new_state = new_state.full()
        # Store results
        test_states[j][i] = new_state
def evolve(time, p, k, nq, jv, iv):
  rho, entangled = test_states[jv][iv], log_neg[jv][iv]
  rho1=rho
  for n in range(int(nq/2)-1):
    rho1 = np.kron(rho1, rho)
  Jx=[]
  op = (get_Jx(nq))
  U = get_U(p,k,nq)
  for t in range(time):
    result = np.trace(np.matmul(rho1, op/(nq/2)))
    rho1 = U @ rho1 @ (U.T).conj()
    Jx.append(np.real(result))
  return np.array(Jx), entangled

class Kickedtop:
  def __init__(self, N_H,dk,pk,lk):
    self.lk=lk
    self.pk=pk
    self.dk=dk
    self.N_H = N_H
    self.r_state = np.zeros(N_H)[self.lk:]
    self.W_out = np.zeros((1, len(self.r_state)))

  def advance_r_state(self,jv,iv):

    self.r_state, self.entangled = evolve(self.N_H, self.pk, self.dk, 4, jv, iv)
    return (self.r_state[self.lk:]), self.entangled

  def v(self):
    return np.dot(self.W_out, self.r_state)

  def train(self, xtrajectory):
    R = np.zeros((len(self.r_state), len(xtrajectory)))
    entangled=np.zeros(len(xtrajectory))
    for i in range(len(xtrajectory)):
      r_state,entangledi = self.advance_r_state(xtrajectory[i][0],xtrajectory[i][1])
      R[:,i] = r_state
      entangled[i]=entangledi

    self.W_out = linear_regression(R, entangled)
    return entangled

  def predict(self, valid):
    prediction = np.zeros(len(valid))
    validation = np.zeros(len(valid))
    for i in range(len(valid)):
      self.r_state,entangledi = self.advance_r_state(valid[i][0],valid[i][1])
      validation[i] = entangledi
      v = (self.v())
      prediction[i] = v
    return prediction, validation

def linear_regression(Rs, trajectory):
    return np.dot(np.linalg.pinv(Rs.T),trajectory)

nonzero_indices = np.nonzero(ent_labels)
nonzero_indices_list = list(zip(nonzero_indices[0], nonzero_indices[1]))
np.random.shuffle((nonzero_indices_list))

zero_indices = np.where(np.array(ent_labels) == 0)
zero_indices_list = list(zip(zero_indices[0], zero_indices[1]))
np.random.shuffle(zero_indices_list)

test_indices =  nonzero_indices_list[500:700]
train_indices = (nonzero_indices_list[00:500])
np.random.shuffle(train_indices)


rmse=[]
for i in np.arange(0,100,5):
 model=Kickedtop(i, 2.8, 3, 0)
 entangled = model.train(train_indices)
 predicted_data, validation_data = model.predict(test_indices)
 rmse.append(np.sqrt(np.mean((predicted_data - validation_data)**2)))
 print(i, rmse[-1])
