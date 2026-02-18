import numpy as np

g = 9.81
ell_1 = 0.247
ell_2 = -0.039
ell_3x = -0.007
ell_3y = -0.007
ell_3z = 0.018
ell_T = 0.355
d = 0.12
m1 = 0.108862
J_1x = 0.000189
J_1y = 0.001953
J_1z = 0.001894
m2 = 0.4717
J_2x = 0.00231
J_2y = 0.003274
J_2z = 0.003416
m3 = 0.1905
J_3x = 0.0002222
J_3y = 0.0001956
J_3z = 0.000027
ts = 0.01
km = 1
beta = 0.001

phi0 = 0.0
theta0 = 0.0
psi0 = 0.0
phidot0 = 0.0
thetadot0 = 0.0
psidot0 = 0.0

##### Chapter 4
# mixing matrices (see end of Chapter 4 in lab manual)
# mixing is a UAV term for taking body forces/torques to individual motor forces
unmixer = np.array([[1.0, 1.0], [d, -d]])  # [F, tau] = unmixer @ [fl, fr]
mixer = np.linalg.inv(unmixer)  # [fl, fr] = mixer @ [F, tau]

