import matplotlib.pyplot as plt
import numpy as np
from case_studies.H_hummingbird.hummingbird_dynamics import HummingbirdDynamics
from case_studies.common.signal_generator import SignalGenerator
from case_studies.common.numeric_integration import rk4_step
from case_studies.H_hummingbird import params as P
from case_studies.H_hummingbird import Visualizer

fr_gen = SignalGenerator(amplitude=3.0, frequency=0.2)
fl_gen = SignalGenerator(amplitude=1.0, frequency=0.3)

x0 = np.zeros(6)

x_hist = [x0]
u_hist = []

system = HummingbirdDynamics()

time = np.arange(0, 20, step=P.ts, dtype=np.float64)

x_prev = x0
u_prev = np.array([0.0, 0.0])

for t in time[1:]:
    u_next = np.array([fl_gen.sin(t), fr_gen.sin(t)])
    y = system.update(u_next)

    x_hist.append(system.state)
    u_hist.append(u_next)

x_hist = np.array(x_hist)
u_hist = np.array(u_hist)

viz = Visualizer(time, x_hist, u_hist)
viz.animate(print_timing=True)
