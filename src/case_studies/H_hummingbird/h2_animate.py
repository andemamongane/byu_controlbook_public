import matplotlib.pyplot as plt
import numpy as np
from case_studies import common, H_hummingbird
P = H_hummingbird.params

# instantiate reference input classes
phi_gen = common.SignalGenerator(amplitude=np.pi / 6, frequency=0.2)
theta_gen = common.SignalGenerator(amplitude=np.pi / 8, frequency=0.3)
psi_gen = common.SignalGenerator(amplitude=np.pi / 2, frequency=0.1)
force_gen = common.SignalGenerator(amplitude=5, frequency=0.5)
tau_gen = common.SignalGenerator(amplitude=1, frequency=0.2)

x0 = np.zeros(6)
x_hist = [x0]
u_hist = []

time = np.arange(0, 20, step=P.ts, dtype=np.float64)
for t in time[1:]:
    x = np.empty(6)
    x[0] = phi_gen.sin(t)
    x[1] = theta_gen.sin(t)
    x[2] = psi_gen.sin(t)
    x[3] = phi_gen.random(t)
    x[4] = theta_gen.random(t)
    x[5] = psi_gen.random(t)

    force = force_gen.sawtooth(t)
    tau = tau_gen.sawtooth(t)
    u_FT = np.array([force, tau])
    u_pwm = P.mixer @ u_FT / P.km

    x_hist.append(x)
    u_hist.append(u_pwm)

x_hist = np.array(x_hist)
u_hist = np.array(u_hist)
viz = H_hummingbird.Visualizer(time, x_hist, u_hist)

viz.animate(print_timing=True)