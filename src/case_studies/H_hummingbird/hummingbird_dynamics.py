# dynamics_tamplate.py

# 3rd-party
import numpy as np

# local (controlbook)

# from case_studies import common, H_hummingbird
# import case_studies.H_hummingbird.params as P
# from case_studies.common.dynamics_base import DynamicsBase
# from case_studies.H_hummingbird import generate_eom

from . import params as P
from ..common.dynamics_base import DynamicsBase
# from . import generate_eom
from . import eom_generated


class HummingbirdDynamics(DynamicsBase):
    def __init__(self, alpha=0.0):
        super().__init__(
            # Initial state conditions
            state0=np.array(
                [P.phi0, P.theta0, P.psi0, P.phidot0, P.thetadot0, P.psidot0]
            ),
            # Input limits [f_l, f_r]
            u_max=np.inf,  # we will saturate later based on input limits
            u_min=0.0,  # no negative forces
            # Time step for integration
            dt=P.ts,
        )
        # See params.py/textbook for details on these parameters

        # Parameter randomization is introduced in Chapter 10, you do not need
        # to include it before then
        # NOTE: Keys must match the parameter names in eom_generated.py functions
        self.eom_params = {
            "m1": self.randomize_parameter(P.m1, alpha),
            "m2": self.randomize_parameter(P.m2, alpha),
            "m3": self.randomize_parameter(P.m3, alpha),
            "J_1x": self.randomize_parameter(P.J_1x, alpha),
            "J_1y": self.randomize_parameter(P.J_1y, alpha),
            "J_1z": self.randomize_parameter(P.J_1z, alpha),
            "J_2x": self.randomize_parameter(P.J_2x, alpha),
            "J_2y": self.randomize_parameter(P.J_2y, alpha),
            "J_2z": self.randomize_parameter(P.J_2z, alpha),
            "J_3x": self.randomize_parameter(P.J_3x, alpha),
            "J_3y": self.randomize_parameter(P.J_3y, alpha),
            "J_3z": self.randomize_parameter(P.J_3z, alpha),
            "ell_1": self.randomize_parameter(P.ell_1, alpha),
            "ell_2": self.randomize_parameter(P.ell_2, alpha),
            "ell_3x": self.randomize_parameter(P.ell_3x, alpha),
            "ell_3y": self.randomize_parameter(P.ell_3y, alpha),
            "ell_3z": self.randomize_parameter(P.ell_3z, alpha),
            "ell_T": self.randomize_parameter(P.ell_T, alpha),
            "d": self.randomize_parameter(P.d, alpha),
            "g": P.g,
        }

        # self.param_vals = [
        #     self.eom_params["m1"],
        #     self.eom_params["m2"],
        #     self.eom_params["m3"],
        #     self.eom_params["J_1x"],
        #     self.eom_params["J_1y"],
        #     self.eom_params["J_1z"],
        #     self.eom_params["J_2x"],
        #     self.eom_params["J_2y"],
        #     self.eom_params["J_2z"],
        #     self.eom_params["J_3x"],
        #     self.eom_params["J_3y"],
        #     self.eom_params["J_3z"],
        #     self.eom_params["ell_1"],
        #     self.eom_params["ell_2"],
        #     self.eom_params["ell_3x"],
        #     self.eom_params["ell_3y"],
        #     self.eom_params["ell_3z"],
        #     self.eom_params["ell_T"],
        #     self.eom_params["d"],
        #     self.eom_params["g"]
        # ]

        p = self.eom_params
        self.km = P.g * (p["m1"] * p["ell_1"] + p["m2"] * p["ell_2"]) / p["ell_T"]
        self.B = P.beta * np.eye(3)  # friction-based terms

    ############################################################################
    # The following 4 methods are where we use the generated functions from
    # SymPy to calculate the terms in the equations of motion: the mass matrix
    # M, the coriolis vector C, the gravity terms dP_dq, and the generalized
    # force vector tau. Each of these methods are just wrappers to call the
    # generated functions with the appropriate parameters.
    # See the end of h3_generate_EL.py for an example of how to call the
    # generated functions.

    # NOTE: IF YOU DID NOT GENERATE THESE FUNCTIONS, YOU WOULD NEED TO
    # IMPLEMENT THE CALCULATION OF THESE TERMS BY HAND.

    # IF YOU DID GENERATE THEM, JUST CALL THE FUNCTIONS HERE BELOW. This would
    # look like the following:
        # eom_generated.calculate_Y(x, **self.eom_params) OR
        # eom_generated.calculate_Y(x, u, **self.eom_params)
    # SEE eom_generated.py FOR THE EXACT FUNCTION NAMES AND ARGUMENTS.


    def calculate_M(self, x):
        M = eom_generated.calculate_M(x, **self.eom_params)
        return M

    def calculate_C(self, x):
        C = eom_generated.calculate_C(x, **self.eom_params)
        return C.flatten()

    def calculate_dP_dq(self, x):
        dP_dq = eom_generated.calculate_dP_dq(x, **self.eom_params)
        return dP_dq.flatten()

    def calculate_tau(self, x, u):
        x_ = x.reshape(-1, 1)
        u_ = u.reshape(-1, 1)
        tau = eom_generated.calculate_tau(x_, u_, **self.eom_params)
        return tau.flatten()

    ############################################################################

    def f(self, x, u):
        """
        Full nonlinear dynamic model `xdot = f(x,u)`.

        Args:
            x (1D numpy array): state vector [q, qdot] or
                [phi, theta, psi, phidot, thetadot, psidot]
            u (1D numpy array): input vector [f_l, f_r] (later will change to PWM values)
        Returns:
            xdot (1D numpy array): time derivative of state vector
                [phidot, thetadot, psidot, phiddot, thetaddot, psiddot]
        """
        # Ensure inputs are flattened for consistent indexing
        # x = x.flatten()
        # u = u.flatten()

        x = x.reshape(-1, 1)
        u = u.reshape(-1, 1)
        
        # Re-label terms for readability
        qdot = x[3:6].reshape((3, 1))
    
        M = self.calculate_M(x)                 # (3,3)
        C = self.calculate_C(x)                 # (3,1)
        dP_dq = self.calculate_dP_dq(x)         # (3,1)
        tau = self.calculate_tau(x, u)          # (3,1)
        Bqdot = self.B @ qdot                   # (3,1)
    
        # Solve for qddot:  M @ qddot = tau - C - dP_dq - B qdot
        rhs = tau.flatten() - C.flatten() - dP_dq.flatten() - Bqdot.flatten()

        #TODO: Find qddot using the equations of motion, then formulate and return xdot
        qddot = np.linalg.solve(M, rhs).flatten()   # (3,)
        xdot = np.concatenate((x[3:6].flatten(), qddot))

        # print()
        # print(M.shape)
        # print(C.shape)
        # print(dP_dq.shape)
        # print(tau.shape)
        # print(Bqdot.shape)
        # print()

        return xdot

    def h(self):
        """
        Output function `y = h(x)`, where x is self.state and does not need to
        be passed in. Here we assume we can measure all the generalized
        coordinates, i.e., position states.

        Returns:
            y (1D numpy array): output vector [phi, theta, psi]
        """
        #TODO: return the measured outputs based on self.state - these would be the first three states
        y = self.state[0:3]#.copy()
        return y

    def update(self, pwm):
        """
        Integrates the system dynamics forward one time step using RK4.
        This version accepts PWM inputs and converts them to forces.

        Args:
            pwm (NDArray[np.float64]): input vector [pwm_l, pwm_r].
        Returns:
            y (NDArray[np.float64]): measured output vector [phi, theta, psi].
        """
        # saturate pwm
        pwm = np.clip(pwm, 0.0, 1.0)

        # convert pwm to motor forces [f_l, f_r]
        u = pwm * self.km

        # call update from parent class (DynamicsBase)
        y = super().update(u)
        return y

# system = HummingbirdDynamics()

# x = np.zeros(6).reshape(-1, 1)
# u = np.zeros(2).reshape(-1, 1)

# print(system.f(x, u))

def rk4_step(fn, x, u, dt):
    """
    Perform a numerical integration step using the 4th-order Runge-Kutta method
    (RK4).

    Args:
        fn: Function that computes the derivative of the state. It should be a
            function of the form fn(x, u) -> xdot.
        x: Current state vector of the system (time-varying over the step).
        u: Control input vector to the system (constant over the step).
            Hint: for observers, the input (labeled here as u) would be the
            measurement y.

    returns:
        x_next: State vector of the system after one time step.
    """
    k1 = fn(x, u)


    k2 = fn(x + k1 * dt / 2, u)
    k3 = fn(x + k2 * dt / 2, u)
    k4 = fn(x + k3 * dt, u)
    xdot = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    x_next = x + xdot * dt
    return x_next


x = np.zeros(6)
u = np.zeros(2)
system = HummingbirdDynamics()
# fn = system.f
dt = 0.001

# k1 = fn(x, u)


# k2 = fn(x + k1 * dt / 2, u)
# k3 = fn(x + k2 * dt / 2, u)
# k4 = fn(x + k3 * dt, u)
# xdot = (k1 + 2 * k2 + 2 * k3 + k4) / 6
# x_next = x + xdot * dt

# print(f"x_next: {x_next}")

# x0 = np.zeros(6)
# u0 = np.zeros(2)
# system = HummingbirdDynamics()

x1 = rk4_step(system.f, x, u, dt)


# system = HummingbirdDynamics()
# def calculate_M(x):
#     M = generate_eom.M_num(x, system.param_vals)
#     return M

# print(calculate_M(np.zeros(6).reshape(-1, 1)))