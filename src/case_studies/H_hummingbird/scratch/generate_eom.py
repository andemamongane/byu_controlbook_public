# generate_eom.py

from sympy import sin, cos
import numpy as np
import case_studies.H_hummingbird.params as p
from case_studies.H_hummingbird.generate_KE import *

g, beta, d, f_l, f_r, ell_T, km = sp.symbols('g beta d f_l f_r ell_T km', real=True)
K = (1/2) * (qdot.T @ M @ qdot)

K = K[0,0]

P = (m1 * g * ell_1 * sp.sin(theta) +
     m2 * g * ell_2 * sp.sin(theta) +
     m3 * g * ell_3z) # + P0 ???

tau = sp.Matrix([[d*(f_l - f_r)],
                 [ell_T*(f_l + f_r)*sp.cos(phi)],
                 [ell_T*(f_l + f_r)*sp.cos(theta)*sp.sin(phi) - d*(f_l - f_r)*sp.sin(theta)]])

C = (sp.diff(M, t) @ qdot -
     (1/2) * (sp.Matrix([qdot.T @ sp.diff(M, phi),
                        qdot.T @ sp.diff(M, theta),
                        qdot.T @ sp.diff(M, psi)])
    @ qdot))

dP_dq = sp.Matrix([sp.diff(P, phi),
                       sp.diff(P, theta),
                       sp.diff(P, psi)])


params_sub = [(g, p.g)]
# print()
# sp.pprint(M)
# print()
M = M.subs(params_sub)
C = C.subs(params_sub)
dP_dq = dP_dq.subs(params_sub)
tau = tau.subs(params_sub)

M = sp.trigsimp(M)
C = sp.trigsimp(C)
dP_dq = sp.trigsimp(dP_dq)
tau = sp.trigsimp(tau)


state = np.array([[phi, theta, psi, qdot[0], qdot[1], qdot[2]]]).T

u = np.array([[f_l, f_r]]).T

params = [m1, m2, m3,
          J1[0,0], J1[1,1], J1[2,2],
          J2[0,0], J2[1,1], J2[2,2],
          J3[0,0], J3[1,1], J3[2,2],
          ell_1, ell_2, ell_3x, ell_3y, ell_3z, ell_T, d, g]

M_num = sp.lambdify((state, *params), np.array(M), modules='numpy')
C_num = sp.lambdify((state, *params), np.array(C), modules='numpy')
dP_dq_num = sp.lambdify((state, *params), np.array(dP_dq), modules='numpy')
tau_num = sp.lambdify((state, u, *params), np.array(tau), modules='numpy')
B_val = p.beta *np.eye(3)

params = {
     "m1": p.m1,
     "m2": p.m2,
     "m3": p.m3,
     "J_1x": p.J_1x,
     "J_1y": p.J_1y,
     "J_1z": p.J_1z,
     "J_2x": p.J_2x,
     "J_2y": p.J_2y,
     "J_2z": p.J_2z,
     "J_3x": p.J_3x,
     "J_3y": p.J_3y,
     "J_3z": p.J_3z,
     "ell_1": p.ell_1,
     "ell_2": p.ell_2,
     "ell_3x": p.ell_3x,
     "ell_3y": p.ell_3y,
     "ell_3z": p.ell_3z,
     "ell_T": p.ell_T,
     "d": p.d,
     "g": p.g,

}
# # Testing
# state = np.array([[0., 0., 0., 0., 0., 0.]]).T
# u = np.array([[0.24468/2, 0.22468/2]]).T
# M_val = M_num(state, **params)
# C_val = C_num(x, param_vals)
# dP_dq_val = dP_dq_num(x, param_vals)
# tau_val = tau_num(x, u, param_vals)
