import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import architecture
import numpy as np
import dynamics as dn
import pickle as pck

with open("rewards/two_gens.pickle", "rb") as handle:
    reward = pck.load(handle)

generator_1 = dn.Generator(1.5, alpha=2)
generator_2 = dn.Generator(1.5, alpha=1)

area = dn.Area(f_set_point=50, m=0.1, d=0.0160, t_g=30, r_d=0.1)
area.set_load(3.15)
area.set_generation(3.0)
area.calculate_delta_f(wiener=True)

z_1 = []
z_2 = []
dz_1 = []
dz_2 = []
ps = []
fs = []
dw = []

tf.reset_default_graph()
graph = tf.train.import_meta_graph("model/model_two_gens.meta")

steps = 100
h_size = 100
a_dof = 2
c_dof = 5
trace = 8
batch = 4
tau = 1
n_vars = 10
arch_per_agent = 4

with tf.Session() as sess:
    graph.restore(sess, "model/model_two_gens")

    agent_1 = architecture.Agent(a_dof, c_dof, h_size, 'agent_1', batch, trace, tau)
    agent_2 = architecture.Agent(a_dof, c_dof, h_size, 'agent_2', batch, trace, tau)

    agent_1.actor.create_op_holder(tf.trainable_variables()[:n_vars], tau)
    agent_2.actor.create_op_holder(tf.trainable_variables()[arch_per_agent * n_vars:(arch_per_agent + 1) * n_vars], tau)

    sess.run(tf.variables_initializer(agent_1.actor.network_params))
    sess.run(tf.variables_initializer(agent_2.actor.network_params))

    sess.run(agent_1.actor.update_parameters)
    sess.run(agent_2.actor.update_parameters)

    st_1 = (np.zeros([1, h_size]), np.zeros([1, h_size]))
    st_2 = (np.zeros([1, h_size]), np.zeros([1, h_size]))

    for i in range(steps):
        z_1.append(generator_1.get_z())
        z_2.append(generator_2.get_z())
        ps.append(area.get_generation())
        fs.append(area.get_frequency())
        dw.append(area.get_delta_f())

        curr_f = area.get_delta_f()
        curr_Z_1 = generator_1.get_z()
        curr_Z_2 = generator_2.get_z()

        a_1, new_st_1 = agent_1.a_actor_operation(sess, np.array([curr_f, curr_Z_1]).reshape(1, a_dof), st_1)
        a_2, new_st_2 = agent_2.a_actor_operation(sess, np.array([curr_f, curr_Z_2]).reshape(1, a_dof), st_2)

        a_1 = a_1[0, 0]
        a_2 = a_2[0, 0]

        dz_1.append(a_1)
        dz_2.append(a_2)

        generator_1.modify_z(a_1)
        generator_2.modify_z(a_2)
        Z = generator_1.get_z() + generator_2.get_z()
        area.calculate_p_g(Z)
        area.calculate_delta_f(wiener=True)

        st_1 = new_st_1
        st_2 = new_st_2

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.rcParams["font.weight"] = 'light'
matplotlib.rc('text', usetex=True)

del matplotlib.font_manager.weight_dict['roman']
matplotlib.font_manager._rebuild()

fig1, ax1 = plt.subplots()
ax1.scatter(np.arange(len(reward)), reward)
ax1.set_xlabel('episodes')
ax1.set_ylabel('cumulative reward [pu]')
ax1.set_xlim(0, 2000)
ax1.set_ylim(0, 1100)
ax1.axhline(1000, c='y')
ax1.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
plt.grid(True)
plt.show()

n = 40
mu = np.array([np.mean(reward[i:i + n]) for i in range(len(reward) - n)])
mu = np.insert(mu, 0, 0., axis=0)
std = np.array([np.std(reward[i:i + n]) for i in range(len(reward) - n)])
std = np.insert(std, 0, 0., axis=0)
upper_bound = mu + 1.96 * std / np.sqrt(n)
lower_bound = mu - 1.96 * std / np.sqrt(n)

fig2, ax2 = plt.subplots()
ax2.plot(np.arange(len(mu)), mu, color='gray', linewidth=0.5)
ax2.fill_between(np.arange(len(upper_bound)), lower_bound, upper_bound, alpha=.1, color='b')
ax2.set_xlabel('episodes')
ax2.set_ylabel('cumulative reward [pu]')
ax2.set_xlim(0, 2000)
ax2.set_ylim(0, 1100)
ax2.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
plt.grid(True)
plt.show()

fig3, ax3 = plt.subplots()
ax3.axhline(3.15, c='y')
ax3.plot(ps, 'b')
ax3.plot(area.p_wiener, 'g', linestyle='--')
ax3.plot(np.sum([np.array(ps), np.array(area.p_wiener[:-1])], axis=0), 'orange', linestyle='-.')
ax3.set_xlabel('time [s]')
ax3.set_ylabel('total power [pu]')
ax3.set_xlim(0, 100)
ax3.legend(['Load', 'Agent-generated', 'Wind-generated', 'Total'])
ax3.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
plt.grid(True)
plt.show()

fig4, ax4 = plt.subplots()
ax4.plot(z_1, 'b')
ax4.plot(z_2, 'r', linestyle='--')
ax4.plot(np.sum([np.array(z_1), np.array(z_2)], axis=0), 'orange', linestyle='-.')
ax4.set_xlabel('time [s]')
ax4.set_ylabel('$z_i$ [pu]')
ax4.set_xlim(0, 100)
ax4.set_ylim(0, 3.5)
ax4.legend(['generator-agent 1', 'generator-agent 2', 'total'])
ax4.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
plt.grid(True)
plt.show()

fig5, ax5 = plt.subplots()
ax5.axhline(np.round(2 * np.pi * 50), c='y')
ax5.plot(np.round(np.array(fs) * 2 * np.pi), 'b')
ax5.set_xlabel('time [s]')
ax5.set_ylabel('$w$ [rad/s]')
ax5.set_xlim(0, 100)
ax5.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
plt.grid(True)
plt.show()

fig6, ax6 = plt.subplots()
ax6.axhline(0, c='y')
ax6.plot(dw)
ax6.set_xlabel('time [s]')
ax6.set_ylabel('dw')
ax6.set_xlim(0, 100)
plt.grid(True)
plt.show()

fig7, ax7 = plt.subplots()
ax7.plot(dz_1, 'b')
ax7.plot(dz_2, 'r', linestyle='--')
ax7.set_xlabel('time [s]')
ax7.set_ylabel('$\Delta z_i$ [pu]')
ax7.set_xlim(0, 100)
ax7.legend(['generator-agent 1', 'generator-agent 2'])
ax7.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
plt.grid(True)
plt.show()

fig8, ax8 = plt.subplots()
ax8.plot(area.p_wiener)
ax8.set_xlabel('time [s]')
ax8.set_ylabel('P [pu]')
ax8.set_xlim(0, 100)
plt.grid(True)
plt.show()
