import matplotlib.pyplot as plt
import matplotlib as mlp
import numpy as np

M = 0.1
D = 0.0160
T_g = 30
R_d = 0.1
rho = 0
a_1 = 2
a_2 = 1
P_c_1 = 1.5
P_c_2 = 1.5
z = P_c_1 + P_c_2
epsilon = 0
sigma_g_1 = .1
sigma_g_2 = .1
sigma_epsilon = .1
P_g = 3.0
P_l = 3.15
delta_omega = 0

powers = []
frequencies = []
cost_1 = []
cost_2 = []
z_1 = []
z_2 = []

for _ in range(100):
    # omega update
    d_delta_omega = (P_g - (1 + rho)*P_l - D*delta_omega)/M
    delta_omega += d_delta_omega

    # generation update
    d_P_g = (-P_g + z - (1/R_d)*delta_omega)/T_g
    P_g += d_P_g

    # control signal update
    d_P_c_1 = -sigma_g_1*(2*a_1*P_c_1 + epsilon)
    P_c_1 += d_P_c_1

    d_P_c_2 = -sigma_g_2*(2*a_2*P_c_2 + epsilon)
    P_c_2 += d_P_c_2

    d_epsilon = sigma_epsilon*(P_c_1 + P_c_2 - (1 + rho)*P_l)
    epsilon += d_epsilon

    z = P_c_1 + P_c_2

    frequencies.append(delta_omega)
    powers.append(P_g)
    cost_1.append(a_1*P_c_1**2)
    cost_2.append(a_2*P_c_2**2)
    z_1.append(P_c_1)
    z_2.append(P_c_2)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.rcParams["font.weight"] = 'light'
mlp.rc('text', usetex=True)

del mlp.font_manager.weight_dict['roman']
mlp.font_manager._rebuild()

plt.plot(np.array(frequencies) + 2*np.pi*50)
plt.xlabel('time [s]')
plt.ylabel('$w$ [rad/s]')
plt.grid(True)
plt.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
plt.xlim(0, 99)
plt.show()

plt.plot(powers)
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('total power [pu]')
plt.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
plt.xlim(0, 99)
plt.show()

plt.plot(z_1)
plt.plot(z_2)
plt.plot(np.array(z_1) + np.array(z_2))
plt.xlabel('time [s]')
plt.ylabel('$z_i$ [pu]')
plt.legend(['generator-agent 1', 'generator-agent 2', 'total'])
plt.grid(True)
plt.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
plt.xlim(0, 99)
plt.show()

plt.plot(cost_1)
plt.plot(cost_2)
plt.xlabel('time [s]')
plt.ylabel('$c_i$ [pu]')
plt.legend(['generator-agent 1', 'generator-agent 2', 'total'])
plt.grid(True)
plt.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
plt.xlim(0, 99)
plt.show()
