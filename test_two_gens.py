import tensorflow as tf
import matplotlib.pyplot as plt
import architecture
import numpy as np
import dynamics as dn
import pickle as pck

with open("two_gens_reward.pickle", "rb") as handle:
    reward = pck.load(handle)

generator_1 = dn.Generator(1.5, alpha=2)
generator_2 = dn.Generator(1.5, alpha=1)
        
area = dn.Area(f_set_point=50, m=0.1, d=0.0160, t_g=30, r_d=0.1)
area.set_load(3.15)
area.set_generation(3.0)
area.calculate_delta_f()

z_1 = []
z_2 = []
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
norm = .2

with tf.Session() as sess:
    
    graph.restore(sess, "model/model_two_gens")

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

        z_1.append(generator_1.get_z())
        z_2.append(generator_2.get_z())
        ps.append(area.get_generation())
        fs.append(area.get_frequency())
        dw.append(area.get_delta_f())

        curr_f = area.get_delta_f()
        curr_Z_1 = generator_1.get_z()
        curr_Z_2 = generator_2.get_z()

        a_1, new_st_1 = agent_1.a_actor_operation(sess, np.array([curr_f, norm * curr_Z_1]).reshape(1, a_dof), st_1)
        a_2, new_st_2 = agent_2.a_actor_operation(sess, np.array([curr_f, norm * curr_Z_2]).reshape(1, a_dof), st_2)
        
        a_1 = a_1[0, 0]
        a_2 = a_2[0, 0]

        generator_1.modify_z(a_1)
        generator_2.modify_z(a_2)
        Z = generator_1.get_z() + generator_2.get_z()
        area.calculate_p_g(Z)
        area.calculate_delta_f()

        st_1 = new_st_1
        st_2 = new_st_2
        
plt.figure(1)
plt.scatter(np.arange(len(reward)), reward)
plt.xlabel('Episodes')
plt.ylabel('Cum. reward per episode')
plt.show()

plt.figure(2)
plt.plot(ps)
plt.plot([3.15]*100)
plt.xlabel('Steps')
plt.ylabel('Power (MW)')
plt.legend(['Total power', 'Power set point'])
plt.show()

plt.figure(3)
plt.plot(z_1)
plt.plot(z_2)
plt.plot(np.sum([np.array(z_1), np.array(z_2)], axis=0))
plt.xlabel('Steps')
plt.ylabel('Control action (Z)')
plt.legend(['Gen 1 secondary action', 'Gen 2 secondary action', 'Total secondary action'])
plt.show()

plt.figure(5)
plt.plot(fs)
plt.plot([50]*100)
plt.xlabel('Steps')
plt.ylabel('Frequency (Hz)')
plt.legend(['System frequency', 'Frequency set point'])
plt.show()

plt.figure(6)
plt.plot(dw)
plt.xlabel('Steps')
plt.ylabel('dw')
plt.show()

plt.figure(7)
z_1_cost = [generator_1.alpha*(x**2) for x in z_1]
z_2_cost = [generator_2.alpha*(x**2) for x in z_2]
plt.plot(z_1_cost)
plt.plot(z_2_cost)
plt.plot(np.sum([np.array(z_1_cost), np.array(z_2_cost)], axis=0))
plt.legend(['Gen 1 cost', 'Gen 2 cost', 'Total cost'])
plt.xlabel('Steps')
plt.ylabel('Operational cost')
plt.show()
