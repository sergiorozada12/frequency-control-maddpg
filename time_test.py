import datetime

import dynamics as dn
import tensorflow as tf
import numpy as np
import architecture

b = np.array([[1, .01], [.01, 1]])
f_m = 1

gen_1 = dn.Generator(1.5, alpha=2)
gen_2 = dn.Generator(1.5, alpha=1)

network_node_1 = dn.NetworkNode(f_set_point=50, m=0.1, d=0.0160, t_g=30, r_d=0.1, idx=0)
network_node_2 = dn.NetworkNode(f_set_point=50, m=0.15, d=0.0180, t_g=30, r_d=0.08, idx=1)

network_node_1.set_load(1.65)
network_node_2.set_load(1.65)

network_node_1.set_generation(1.5)
network_node_2.set_generation(1.5)

network = dn.Network([network_node_1, network_node_2], b, f_m)

network_node_1.set_true_load(network.get_true_load(network_node_1.idx))
network_node_2.set_true_load(network.get_true_load(network_node_2.idx))

network_node_1.calculate_delta_f()
network_node_2.calculate_delta_f()

tf.reset_default_graph()
graph = tf.train.import_meta_graph("model/model_two_gens_network_cost.meta")


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

    agent = architecture.Agent(a_dof, c_dof, h_size, 'agent_1', batch, trace, tau)
    agent.actor.create_op_holder(tf.trainable_variables()[:n_vars], tau)

    sess.run(tf.variables_initializer(agent.actor.network_params))

    sess.run(agent.actor.update_parameters)

    agent.a_actor_operation(sess, np.array([.0, .0]).reshape(1, 2), (np.zeros([1, 100]), np.zeros([1, 100])))

    start = datetime.datetime.now()
    for _ in range(10):
        _, _ = agent.a_actor_operation(sess, np.array([.0, .0]).reshape(1, 2), (np.zeros([1, 100]), np.zeros([1, 100])))
    end = datetime.datetime.now()

print((end - start).total_seconds())

