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

def get_uq(jx, jy, jz, p, k0 = 3.0):
    u1= expm(-1j* p* jy)
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
  def __init__(self, theta, phi,k, p, N_H):
    self.k=k
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

    u = ((u - min(x_list))/(max(x_list) - min(x_list)))

    state = evolve(self.N_H, 3.6 + self.k*u, j, jx, jy, jz, d=0)

    self.r_state = state[:,0]
    return self.r_state

  def v(self):
    return np.dot(self.W_out, self.r_state)

  def train(self, xtrajectory):
    R = np.zeros((len(self.r_state)*self.p, xtrajectory.shape[0]-self.p))
    self.stack=[]
    for p in range(len(xtrajectory)):
        self.stack.append(self.advance_r_state(xtrajectory[p]))
    for i in range(len(xtrajectory)-self.p):
      R[:, i] = np.hstack( tuple((self.stack[i+(self.p-1)-p])*((self.p-p)/self.p) for p in reversed(range(self.p)) ))
    def linear_regression(Rs, trajectory, beta=1*0.00041):
      Rt = np.transpose(Rs)
      inverse_part = np.linalg.inv(np.dot(Rs, Rt) + beta * np.identity(Rs.shape[0]))
      return np.dot(np.dot(trajectory.T, Rt), inverse_part)
    self.W_out = linear_regression(R, xtrajectory[self.p:])


  def predict(self, steps):
    prediction = np.zeros(steps)
    for i in range(steps):
      self.r_state= np.hstack( tuple((self.stack[int(data_length*training_percentage-1) + i -p])*((self.p-p)/self.p) for p in reversed(range(self.p)) ))
      v = self.v()
      prediction[i] = v
      self.stack.append(self.advance_r_state(prediction[i]))
    return prediction

def lorenz(t, x, y, z, sigma, rho, beta):
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x*y - beta*z
    return dxdt, dydt, dzdt

def rk4(t, x, y, z, dt, sigma, rho, beta):
    f1x, f1y, f1z = lorenz(t, x, y, z, sigma, rho, beta)

    k1x = dt * f1x
    k1y = dt * f1y
    k1z = dt * f1z

    f2x, f2y, f2z = lorenz(t + 0.5*dt, x + 0.5*k1x, y + 0.5*k1y, z + 0.5*k1z, sigma, rho, beta)

    k2x = dt * f2x
    k2y = dt * f2y
    k2z = dt * f2z

    f3x, f3y, f3z = lorenz(t + 0.5*dt, x + 0.5*k2x, y + 0.5*k2y, z + 0.5*k2z, sigma, rho, beta)

    k3x = dt * f3x
    k3y = dt * f3y
    k3z = dt * f3z

    f4x, f4y, f4z = lorenz(t + dt, x + k3x, y + k3y, z + k3z, sigma, rho, beta)

    k4x = dt * f4x
    k4y = dt * f4y
    k4z = dt * f4z

    x += (k1x + 2*k2x + 2*k3x + k4x) / 6.0
    y += (k1y + 2*k2y + 2*k3y + k4y) / 6.0
    z += (k1z + 2*k2z + 2*k3z + k4z) / 6.0

    return x, y, z

x1, y1, z1 = 1, 1, 1
sigma, rho, beta = 10, 28, 8/3
t, dt = 0, 0.01
#integration loop
x_list = []
y_list = []
z_list = []

for i in range(20000):
    x, y, z = rk4(t, x1, y1, z1, dt, sigma, rho, beta)
    x_list.append(x)
    y_list.append(y)
    z_list.append(z)
    x1, y1, z1 = x, y, z
    t += dt

data_length = 6000
training_percentage = 0.834

training_datax = np.array(x_list[1000:1000+int(data_length*training_percentage)])
valid_datax = np.array(x_list[1000+int(data_length*training_percentage):1000 + data_length])
x_list = np.concatenate((training_datax, valid_datax))

model=Kickedtop(2.36,4.89,0.2,64,137)
model.train(training_datax)
predicted_data = model.predict(len(valid_datax))
print(np.sqrt(np.mean((predicted_data[:500] - valid_datax[:500])**2)))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
fig, ax = plt.subplots(figsize=(12, 4.6))
font = {'size'   : 15}

mpl.rc('font', **font)
ax.grid(True,which='major',axis='both',alpha=0.3)


timesteps =  [*range(5000,5000+len(predicted_data))]
ax.plot(timesteps, valid_datax,linestyle='dashed', label="x_actual", lw=2.1)
ax.plot(timesteps, predicted_data, label="x_predicted", color='orange', lw=2.1)
ax.set_ylabel("X Component",fontsize=15)
ax.set_xlabel("Time Steps",fontsize=15)
ax.legend()
fig.tight_layout()
plt.show()
