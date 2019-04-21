import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import architecture
import numpy as np
import dynamics as dn
import pickle as pck

with open("rewards/two_gens_network_reward.pickle", "rb") as handle:
    reward = pck.load(handle)

b = np.array([[1, .01], [.01, 1]])

gen_1 = dn.Generator(1.5, alpha=2)
gen_2 = dn.Generator(1.5, alpha=1)

network_node_1 = dn.NetworkNode(f_set_point=50, m=0.1, d=0.0160, t_g=30, r_d=0.1, idx=0)
network_node_2 = dn.NetworkNode(f_set_point=50, m=0.15, d=0.0180, t_g=30, r_d=0.08, idx=1)

network_node_1.set_load(1.5 + (- 0.25 + np.random.rand() / 2))
network_node_2.set_load(1.5 + (- 0.25 + np.random.rand() / 2))

network_node_1.set_generation(1.5)
network_node_2.set_generation(1.5)

network = dn.Network([network_node_1, network_node_2], b)

network_node_1.set_true_load(network.get_true_load(network_node_1.idx))
network_node_2.set_true_load(network.get_true_load(network_node_2.idx))

network_node_1.calculate_delta_f()
network_node_2.calculate_delta_f()

f_1 = []
f_2 = []
z_1 = []
z_2 = []
dw_1 = []
dw_2 = []
p_1 = []
p_2 = []

tf.reset_default_graph()
graph = tf.train.import_meta_graph("model/model_two_gens_network.meta")

steps = 100
h_size = 100
a_dof = 2
c_dof = 6
trace = 8
batch = 4
tau = 1
n_vars = 10
arch_per_agent = 4

with tf.Session() as sess:
    
    graph.restore(sess, "model/model_two_gens_network")

    agent_1 = architecture.Agent(a_dof, c_dof, h_size, 'agent_1', batch, trace, tau)
    agent_2 = architecture.Agent(a_dof, c_dof, h_size, 'agent_2', batch, trace, tau)
    
    agent_1.actor.create_op_holder(tf.trainable_variables()[:n_vars], tau)
    agent_2.actor.create_op_holder(tf.trainable_variables()[arch_per_agent*n_vars:(arch_per_agent+1)*n_vars], tau)
    
    sess.run(tf.variables_initializer(agent_1.actor.network_params))
    sess.run(tf.variables_initializer(agent_2.actor.network_params))
    
    sess.run(agent_1.actor.update_parameters)
    sess.run(agent_2.actor.update_parameters)

    st_1 = (np.zeros([1, h_size]), np.zeros([1, h_size]))
    st_2 = (np.zeros([1, h_size]), np.zeros([1, h_size]))
    
    for i in range(steps):

        z_1.append(gen_1.get_z())
        z_2.append(gen_2.get_z())
        p_1.append(network_node_1.get_generation())
        p_2.append(network_node_2.get_generation())
        f_1.append(network_node_1.get_frequency())
        f_2.append(network_node_2.get_frequency())
        dw_1.append(network_node_1.get_delta_f())
        dw_2.append(network_node_2.get_delta_f())

        curr_f_1 = network_node_1.get_delta_f()
        curr_f_2 = network_node_2.get_delta_f()

        curr_Z_1 = gen_1.get_z()
        curr_Z_2 = gen_2.get_z()

        a_1, new_st_1 = agent_1.a_actor_operation(sess, np.array([curr_f_1, curr_Z_1]).reshape(1, a_dof), st_1)
        a_2, new_st_2 = agent_2.a_actor_operation(sess, np.array([curr_f_2, curr_Z_2]).reshape(1, a_dof), st_2)
        
        a_1 = a_1[0, 0]
        a_2 = a_2[0, 0]

        gen_1.modify_z(a_1)
        gen_2.modify_z(a_2)

        network_node_1.calculate_p_g(gen_1.get_z())
        network_node_2.calculate_p_g(gen_2.get_z())

        network_node_1.calculate_nu()
        network_node_2.calculate_nu()

        network_node_1.set_true_load(network.get_true_load(network_node_1.idx))
        network_node_2.set_true_load(network.get_true_load(network_node_2.idx))

        network_node_1.calculate_delta_f()
        network_node_2.calculate_delta_f()

        st_1 = new_st_1
        st_2 = new_st_2

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.rcParams["font.weight"] = 'light'

del matplotlib.font_manager.weight_dict['roman']
matplotlib.font_manager._rebuild()

fig1, ax1 = plt.subplots()
ax1.scatter(np.arange(len(reward)), reward)
ax1.set_xlabel('episodes')
ax1.set_ylabel('cumulative reward [pu]')
ax1.set_xlim(0, 20000)
ax1.set_ylim(0, 22500)
ax1.axhline(20000, c='y')
ax1.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
plt.grid(True)
plt.show()

n = 40
mu = np.array([np.mean(reward[i:i+n]) for i in range(len(reward)-n)])
mu = np.insert(mu, 0, 0., axis=0)
std = np.array([np.std(reward[i:i+n]) for i in range(len(reward)-n)])
std = np.insert(std, 0, 0., axis=0)
upper_bound = mu + 1.96*std/np.sqrt(n)
lower_bound = mu - 1.96*std/np.sqrt(n)

fig2, ax2 = plt.subplots()
ax2.plot(np.arange(len(mu)), mu, color='gray', linewidth=0.5)
ax2.fill_between(np.arange(len(upper_bound)), lower_bound, upper_bound, alpha=.1, color='b')
ax2.set_xlabel('episodes')
ax2.set_ylabel('cumulative reward [pu]')
ax2.set_xlim(0, 20000)
ax2.set_ylim(0, 20000)
ax2.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
plt.grid(True)
plt.show()

fig3, ax3 = plt.subplots()
ax3.plot(p_1, 'b')
ax3.plot(p_2, 'r', linestyle='--')
ax3.set_xlabel('time [s]')
ax3.set_ylabel('total power [pu]')
ax3.set_xlim(0, 100)
ax3.legend(['generator-agent 1', 'generator-agent 2'])
ax3.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
plt.grid(True)
plt.show()

fig4, ax4 = plt.subplots()
ax4.plot(z_1, 'b')
ax4.plot(z_2, 'r', linestyle='--')
ax4.set_xlabel('time [s]')
ax4.set_ylabel('z [pu]')
ax4.set_xlim(0, 100)
ax4.legend(['generator-agent 1', 'generator-agent 2'])
ax4.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
plt.grid(True)
plt.show()

fig5, ax5 = plt.subplots()
ax5.plot(f_1, 'b')
ax5.plot(f_2, 'r', linestyle='--')
ax5.set_xlabel('time [s]')
ax5.set_ylabel('f (Hz)')
ax5.set_xlim(0, 100)
ax5.legend(['generator-agent 1', 'generator-agent 2'])
ax5.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
plt.grid(True)
plt.show()

fig6, ax6 = plt.subplots()
ax6.plot(dw_1, 'b')
ax6.plot(dw_2, 'r', linestyle='--')
ax6.set_xlabel('time [s]')
ax6.set_ylabel('dw')
ax6.legend(['generator-agent 1', 'generator-agent 2'])
ax6.set_xlim(0, 100)
plt.grid(True)
plt.show()
