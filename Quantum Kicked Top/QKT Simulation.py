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

def get_uq(jx, jy, jz, k0):
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


def evolve(theta,phi,time, k, d=0.0):
    global j
    # print("theta ", theta, "phi ", phi, "delta ", d, "kick ", k)
    expect = np.zeros((time,3))
    jj_vector = np.zeros(int(2*j+1), dtype=complex)
    init = np.zeros(int(2*j+1), dtype=complex)
    init[-1] = 1
    rotate = expm(-1j*theta*(jx*np.sin(phi) - jy*np.cos(phi)))
    jj_vector = np.matmul(rotate,init)
    # rho = np.outer(jj_vector,np.conj(jj_vector))
    # print(rho)

    uq = get_uq(jx, jy, jz, k)

    for t in range(time):
        jj_vector = np.matmul(uq,jj_vector)
        expect[t,:] = get_expectations(jj_vector, jx, jy, jz, j)
        # rho = uq @ rho @ (uq.T).conj()
        # expect[t,:] = expectation_dm(rho, jx, jy, jz, j)
    return expect

j=3/2
runtime=100
jx,jy,jz = get_operator(j)
x=np.zeros(runtime)
z=np.zeros(runtime)
state= evolve(2.25,0.63,runtime,3)
x=state[:,0]
y=state[:,1]
z=state[:,2]

fig = plt.figure(figsize=(12, 3))
ax1 = fig.add_subplot(111)
plt.figure(1)
plt.plot(np.arange(runtime), state[:,0], markersize=3, linewidth=0.5)

plt.xlabel('Time')
plt.ylabel('X')
plt.show()
