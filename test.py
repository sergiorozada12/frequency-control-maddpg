import tensorflow as tf
import matplotlib.pyplot as plt
import architecture
import numpy as np
import dynamics as dn
import pickle as pck

with open("reward.pickle", "rb") as handle:
    reward = pck.load(handle)

generator_1 = dn.Node(1.5, alpha=2)
generator_2 = dn.Node(1.5, alpha=1)
        
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
graph = tf.train.import_meta_graph("model/model.meta")

steps = 100
h_size = 100

with tf.Session() as session:
    
    graph.restore(session, "model/model")
    
    lstm_actor_1 = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
    actor_1 = architecture.Actor(h_size, lstm_actor_1, 'actor_1_test', len(tf.trainable_variables()))
    actor_1.create_op_holder(tf.trainable_variables(), 1)
    
    n_params = len(actor_1.network_params)
    
    lstm_actor_2 = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
    actor_2 = architecture.Actor(h_size, lstm_actor_2, 'actor_2_test', len(tf.trainable_variables()))
    actor_2.create_op_holder(tf.trainable_variables()[n_params*4:], 1)

    init_1 = tf.variables_initializer(actor_1.network_params)
    init_2 = tf.variables_initializer(actor_2.network_params)
    
    session.run(init_1)
    session.run(init_2)
    
    session.run(actor_1.update_parameters)
    session.run(actor_2.update_parameters)

    state_1 = (np.zeros([1, h_size]), np.zeros([1, h_size]))
    state_2 = (np.zeros([1, h_size]), np.zeros([1, h_size]))
    
    for i in range(steps):

        z_1.append(generator_1.get_z())
        z_2.append(generator_2.get_z())
        ps.append(area.get_generation())
        fs.append(area.get_frequency())
        dw.append(area.get_delta_f())

        current_f = area.get_delta_f()
        
        a_1, new_state_1 = session.run([actor_1.a, actor_1.rnn_state],
                                       feed_dict={actor_1.f: np.array(current_f).reshape(1, 1),
                                                  actor_1.p: np.array(generator_1.get_z()).reshape(1, 1),
                                                  actor_1.state_in: state_1,
                                                  actor_1.batch_size: 1, actor_1.train_length: 1})
        a_2, new_state_2 = session.run([actor_2.a, actor_2.rnn_state],
                                       feed_dict={actor_2.f: np.array(current_f).reshape(1, 1),
                                                  actor_2.p: np.array(generator_2.get_z()).reshape(1, 1),
                                                  actor_2.state_in: state_2,
                                                  actor_2.batch_size: 1,
                                                  actor_2.train_length: 1})
        a_1 = a_1[0, 0]
        a_2 = a_2[0, 0]

        generator_1.modify_z(a_1)
        generator_2.modify_z(a_2)
        Z = generator_1.get_z() + generator_2.get_z()
        area.calculate_p_g(Z)
        area.calculate_delta_f()

        state_1 = new_state_1
        state_2 = new_state_2
        
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
