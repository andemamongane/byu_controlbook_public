# %%
################################################################################
# This file is meant to be run interactively with VSCode's Jupyter extension.
# It works as a regular Python script as well, but the printed results will not
# display as nicely.
################################################################################

# %% [markdown]
# # Lab H.3: Equations of Motion
# ### Load H.2

# %%
# It is generally not recommended to import * (everything) from any module or
# package, but in this case we are essentially extending the previous lab file...
# they would normally be in the same file, but we are breaking them up to have
# one file per lab assignment.
import case_studies.H_hummingbird.params as P
from case_studies.H_hummingbird.generate_KE import *
from case_studies.common import sym_utils as su


# This makes it so printing from su only happens when running this file directly
su.enable_printing(__name__ == "__main__")

# %%[markdown]
# # Part 1: Calculate Kinetic Energy, Potential Energy
K = 1.0 / 2.0 * qdot.T @ M @ qdot  # calculate the kinetic energy
K = K[0, 0]


# TODO: define symbols needed for potential energy and the RHS 
g, beta, d, f_l, f_r, ell_T, km = sp.symbols("g beta d f_l f_r ell_T k_m")

# TODO now calculate the potential energy "P"
P = m1 * g * ell_1 * sp.sin(theta) + m2 * g * ell_2 * sp.sin(theta) + m3 * g * ell_3z
# su.printeq("P", P)


# # Part 2: Calculate C, dP/dq, and tau
tau = sp.Matrix(
    [
        d * (f_l - f_r),
        ell_T * (f_l + f_r) * sp.cos(phi),
        ell_T * (f_l + f_r) * sp.cos(theta) * sp.sin(phi)
        - d * (f_l - f_r) * sp.sin(theta),
    ]
)


# Method 2: derive tau manually using virtual work

# Define position vector from origin to where rotor forces act in body frame
r_fl_in_b = sp.Matrix([ell_T, -d, 0])
r_fr_in_b = sp.Matrix([ell_T, d, 0])

# Express position vectors in world/inertial frame so they become functions of q
# R1 was calculated in generate_KE.py and is the rotation from body to world frame
r_fl_in_w = R1 @ r_fl_in_b
r_fr_in_w = R1 @ r_fr_in_b

# Define rotor forces in body frame (they act purely in z direction)
fl_in_b = sp.Matrix([0, 0, -f_l])
fr_in_b = sp.Matrix([0, 0, -f_r])

# Express rotor forces in world/inertial frame
fl_in_w = R1 @ fl_in_b
fr_in_w = R1 @ fr_in_b

# Now we can find tau by doing the dot product of each applied force (fl and fr)
# in the direction of each generalized coordinate (d r_f)/(d q[i])
tau = sp.zeros(3, 1)
for i in range(len(q)):
    tau_i = fl_in_w.T @ r_fl_in_w.diff(q[i]) + fr_in_w.T @ r_fr_in_w.diff(q[i])
    tau_i = sp.trigsimp(tau_i[0])
    tau[i] = tau_i
# END SOLUTION

# Group terms together for readability (this is to help with checking the result
# but is not strictly necessary)
tau[-1] = sp.collect(tau[-1], [sp.sin(theta), sp.sin(phi) * sp.cos(theta)])  # type: ignore[operator]
tau[-1] = sp.collect(tau[-1], [f_l + f_r, f_l - f_r])

su.printeq("tau", tau)


# ### Coriolis Matrix (C):
Mdot = M.diff(t)

# Print Mdot with some terms substituted to make it easier to read
Mdot22 = Mdot[1, 1]
Mdot23 = Mdot[1, 2]
Mdot33 = Mdot[2, 2]
print_subs = {
    Mdot22: sp.Symbol("Mdot_22"),
    Mdot33: sp.Symbol("Mdot_33"),
    Mdot23: sp.Symbol("Mdot_23"),
}
su.printeq("Mdot", Mdot.subs(print_subs))

# Wrap Mdot inside of sp.Matrix call so that individual elements can be modified
Mdot = sp.Matrix(Mdot)
Mdot22 = sp.collect(Mdot22, 2 * sp.sin(phi) * sp.cos(phi) * qdot[0])  # type: ignore[operator]
su.printeq("Mdot_22", Mdot22)
Mdot[1, 1] = Mdot22  # replace with simplified version

Mdot23 = sp.trigsimp(sp.factor(Mdot23))
su.printeq("Mdot_23", Mdot23)
Mdot[1, 2] = Mdot23  # replace with simplified version
Mdot[2, 1] = Mdot23  # Mdot is symmetric

Mdot33 = sp.factor(Mdot33)
Mdot33 = sp.collect(Mdot33, [qdot[0], qdot[1] * sp.sin(theta)])
Mdot33 = sp.expand_trig(sp.trigsimp(sp.collect(Mdot33, Mdot33.free_symbols)))
su.printeq("Mdot_33", Mdot33)
Mdot[2, 2] = Mdot33  # replace with simplified version


# %%
# TODO: calculate the partial derivatives of M with respect to each generalized
# coordinate and verify with lab manual (if they do not match perfectly, you can
# try using sp.trigsimp(), sp.simplify(), sp.collect(), etc., but you may want
# to just move on and use numerical comparisons in the testDynamics.py file to
# verify your matrices are correct)

dM_dphi = M.diff(phi)
dM_dphi = sp.simplify(dM_dphi)

dM_dtheta = M.diff(theta)

# substitute N33 just for printing to match the lab manual
N33 = dM_dtheta[2, 2]
print_subs = {N33: sp.Symbol("N_33")}
su.printeq("dM/dθ", dM_dtheta.subs(print_subs))

N33 = sp.trigsimp(sp.collect(N33, N33.free_symbols))
N33 = sp.trigsimp(sp.collect(N33, N33.free_symbols))  # repeat simplifies further
su.printeq("N_33", N33)

# wrap dM_dtheta inside of sp.Matrix call so that individual elements can be modified
dM_dtheta = sp.Matrix(dM_dtheta)
dM_dtheta[2, 2] = N33

dM_dpsi = M.diff(psi)

qdot_Mdiff = sp.Matrix([qdot.T @ dM_dphi, qdot.T @ dM_dtheta, qdot.T @ dM_dpsi])
C = Mdot @ qdot - half * qdot_Mdiff @ qdot
dP_dq = P.diff(q)

# # Step 3: Create functions for M, C, dP/dq, and tau
# TODO: now you can either run the code below to generate functions to calculate
# M, C, tau, and dP_dq (where you can import the generated functions into your 
# dynamics class), or you can hard-code the results you have found above into the 
# hummingbirdDynamics.py file directly.

# Define state and input vectors for function generation
phidot, thetadot, psidot = qdot
state = sp.Matrix([phi, theta, psi, phidot, thetadot, psidot])
u = sp.Matrix([f_l, f_r])

# Create a list of all parameters needed for the functions
# TODO: you may need to modify this to match the variable names you created above
sys_params = list(
    [m1, m2, m3, J_1x, J_1y, J_1z, J_2x, J_2y, J_2z, J_3x, J_3y, J_3z]
    + [ell_1, ell_2, ell_3x, ell_3y, ell_3z, ell_T, d, g]
)

# Everything below this point will only run if this file is executed directly,
# not if it is imported as a module.
if __name__ == "__main__":
    from case_studies import H_hummingbird

    su.write_eom_to_file(
        state, u, sys_params, H_hummingbird, M=M, C=C, dP_dq=dP_dq, tau=tau
    )
