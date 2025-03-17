import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import unitary_group
from scipy.linalg import sqrtm,expm,logm
from scipy.special import factorial
from scipy.signal import find_peaks
from scipy.special import expit

def get_operator(j):
    n = int(2*j+1)
    jplus= np.zeros((n, n))
    jminus= np.zeros((n, n))

    for m2 in np.arange(-j,j+1,1):
        for m1 in np.arange(-j,j+1,1):
            if m1== m2 + 1:
                jminus[int(m1+j)][int(m2+j)]=np.sqrt((j*(j+ 1)- m2*(m2+ 1)))

    for m1 in np.arange(-j,j + 1,1):
        for m2 in np.arange(-j,j+1,1):
            if m1== m2-1:
                jplus[int(m1+j)][int(m2+j)]=np.sqrt((j*(j+ 1)- m2*(m2-1)))

    jx= (jplus + jminus)/2
    jy= (jplus - jminus)/(2*1j)
    jz = np.diag(np.arange(j, -j-1, -1))
    # jz= ((np.matmul(jx,jy)-np.matmul(jy,jx))/(1j))

    return jx, jy, jz

def get_uq(jx, jy, jz, k0 = 3.0):
    u1= expm(-1j*np.pi/2* jy)
    u2= expm((-1j*k0/(2*j))*np.matmul(jz,jz))
    uq=u1@u2
    return uq


def get_expectations(jj_vector, jx, jy, jz, j):
    """
    Calculate the expectation values of the angular momentum operators.

    Parameters:
    jj_vector (np.array): The state vector.
    jx, jy, jz (np.array): The angular momentum operators.
    j (float): The spin of the system.

    Returns:
    float, float, float: The expectation values of jx, jy, and jz.
    """
    expct_jx = np.matmul(np.conj(jj_vector),np.matmul(jx/j,jj_vector))
    expct_jy = np.matmul(np.conj(jj_vector),np.matmul(jy/j,jj_vector))
    expct_jz = np.matmul(np.conj(jj_vector),np.matmul(jz/j,jj_vector))

    return expct_jx.real, expct_jy.real, expct_jz.real

def expectation(rho, jx, jy, jz, j):
    return np.trace(np.matmul(rho,jx/j)), np.trace(np.matmul(rho,jy/j)), np.trace(np.matmul(rho,jz/j))

j=3/2
jx,jy,jz = get_operator(j)
class Kickedtop:
  def __init__(self, theta, phi, N_H,p):
    self.p=p
    self.theta = theta                                    # list of starting point theta position
    self.phi = phi                                      # list of starting point phi position
    self.N_H = N_H
    self.r_state = np.zeros(N_H)
    self.W_out = np.zeros((1, len(self.r_state)))
    self.jj_vector = np.zeros(int(2*j+1), dtype=complex)
    self.init = np.zeros(int(2*j+1), dtype=complex)
    self.init[-1] = 1
    self.rotate = expm(-1j*self.theta*(jx*np.sin(self.phi) - jy*np.cos(self.phi)))

  def advance_r_state(self, u):

    self.jj_vector = np.matmul(self.rotate,self.init)

    def evolve(time, k, j, jx, jy, jz, d=0.0):
      expect = np.zeros((time,3))
      uq = get_uq(jx, jy, jz, k)
      for t in range(time):
        self.jj_vector = np.matmul(uq,self.jj_vector)
        expect[t,:] = get_expectations(self.jj_vector, jx, jy, jz, j)
      return expect

    u = ((u - min(input))/(max(input) - min(input)))

    state = evolve(self.N_H, 0 + self.p*u, j, jx, jy, jz, d=0)

    self.r_state = state[:,0]
    return self.r_state

  def v(self):
    return np.dot(self.W_out, self.r_state)

  def train(self, xtrajectory, ytrajectory):
    R = np.zeros((len(self.r_state), xtrajectory.shape[0]))
    for i in range(xtrajectory.shape[0]):
      u = xtrajectory[i]
      self.advance_r_state(u)
      R[:,i] = self.r_state
    self.W_out = linear_regression(R, ytrajectory)
    return R

  def predict(self, valid):
    prediction = np.zeros(len(valid))
    for i in range(len(valid)):
      u = valid[i]
      self.advance_r_state(u)
      v = self.v()
      prediction[i] = v
    return prediction

def linear_regression(R, trajectory, beta=0):
  return np.dot(np.linalg.pinv(R.T),trajectory)

    # sigmoid function
def sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

def poly(x):
    return (x - 3)*(x-2)*(x-1)*x*(x+2)*(x+1)*(x + 3)

input = np.linspace(-3, 3, 600)
np.random.shuffle(input)
output = poly(input)

data_length = len(input)
training_percentage = 0.83

training_datax = np.array(input[:int(training_percentage*data_length)])
training_datay = np.array(output[:int(training_percentage*data_length)])
valid_datax = np.array(input[int(training_percentage*data_length):])
valid_datay = np.array(output[int(training_percentage*data_length):])

model=Kickedtop(2.36, 4.89, 32, 5.5)
Ri=model.train(training_datax, training_datay)
predicted_data = model.predict(valid_datax)
np.sqrt(np.mean((valid_datay - predicted_data)**2))

# prompt: plot valid datay with predicted data, sorted accordin to input

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
fig, ax = plt.subplots(figsize=(8, 6))
font = {'size'   : 12}

mpl.rc('font', **font)

ax.grid(True,which='major',axis='both',alpha=0.3)

# Sort the data based on input values
sorted_indices = np.argsort(valid_datax)
valid_datax_sorted = valid_datax[sorted_indices]
predicted_data_sorted = predicted_data[sorted_indices]
sorted_indices1 = np.argsort(input)
input_sorted = input[sorted_indices1]
output_sorted = output[sorted_indices1]

# Plot the data
plt.plot(input_sorted, output_sorted, label='Target Output')
plt.plot(valid_datax_sorted, predicted_data_sorted, ".", label='Predicted Output')
plt.xlabel('x',fontsize=18)
plt.ylabel('f(x)',fontsize=18)

plt.legend()
plt.show()

#plot Ri matrix, evolution of readout with tim
for i in range(0,Ri.shape[1],150):
  plt.plot(Ri[:, i], label=f"Column {i+1}")

plt.xlabel("Time-steps")
plt.ylabel(r'$\langle \text{J}_x \rangle$' "Values")

plt.show()

#fourier transform of readouts
count_freq_z = np.zeros(len(training_datax))[::150]
for i in range(0,len(training_datax),150):
 plt.figure(1)
 fft_2 = np.fft.fft(Ri[:, i])
    # Compute the frequencies for the FFT
 freq = np.fft.fftfreq(Ri[:, i].shape[0])
    # Plot the absolute value of the FFT

 plt.plot(freq, np.abs(fft_2), marker='*')
 plt.xlabel('Frequency')
 plt.ylabel('Amplitude')

 peaks, _ = find_peaks(np.abs(fft_2))
 count_freq_z[int(i/150)] = len(peaks)

plt.figure(2)
plt.plot([*range(0,len(training_datax),150)], count_freq_z, marker='*')
plt.xlabel('input initialization')
plt.ylabel('peak count in fourier spectrum of Sy timeseries')
plt.ylim(-1, max(count_freq_z)+1)
plt.show()
