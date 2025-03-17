import numpy as np
import matplotlib.pyplot as plt

#Evolving function
def evolve(theta, phi, p, k):
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)

    dx = (x*np.cos(p) + z*np.sin(p)) * np.cos(k*(z*np.cos(p) - x*np.sin(p))) - y*np.sin(k*(z*np.cos(p) - x*np.sin(p)))
    dy = (x*np.cos(p) + z*np.sin(p)) * np.sin(k*(z*np.cos(p) - x*np.sin(p))) + y*np.cos(k*(z*np.cos(p) - x*np.sin(p)))
    dz = -x*np.sin(p) + z*np.cos(p)

    dtheta = np.arccos(dz)
    dphi = np.arctan2(dy,dx)
    return dtheta, dphi

#Initial parameters
steps = 200
p0 = np.pi/2
k0 = 3
Ntheta = 30
Nphi = 60
thetas = np.zeros((Ntheta, Nphi, steps))
phis = np.zeros((Ntheta, Nphi, steps))
for i in range(Ntheta):
    thetas[i,:,0] = [i*np.pi/Ntheta ]
    for j in range(Nphi):
        phis[:,j,0] = [2*j*np.pi/Nphi]

#Main function
for i in range(Ntheta):
    for j in range(Nphi):
        theta0 = thetas[i,j,0]
        phi0 = phis[i,j,0]
        for n in range(1, steps):
            theta0, phi0 = evolve(theta0, phi0, p0, k0)
            thetas[i,j,n] = theta0
            if phi0 < 0:
                phis[i,j,n] = phi0+2*np.pi
            else:
                phis[i,j,n] = phi0

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
phis = np.reshape(phis, (Ntheta*Nphi*steps))
thetas = np.reshape(thetas, (Ntheta*Nphi*steps))
plt.figure(figsize=(10, 6))
plt.plot(phis,thetas, 'b.', ms = 0.1)

plt.tick_params(axis='x',which='major',direction='out')
plt.tick_params(axis='y',which='major',direction='out')
plt.xlabel('$\phi$',fontsize=15)
plt.xlim(0, 2*np.pi)
plt.ylabel('$\Theta$',fontsize=15)
plt.ylim(0,np.pi)
plt.tight_layout()
plt.show()
