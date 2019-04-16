import dynamics as dn
import rl
import tensorflow as tf
import numpy as np
import architecture
import pickle as pck

""" MODEL DEFINITION"""

# Parameters
gamma = 0.9
tau = 0.001
epsilon = 0.99
episodes = 10000
steps = 100
cum_r = 0
trace = 8
batch = 4
n_var = 12
buffer_size = 1000
p_tot = 0
cum_r_list = []
b = np.array([[1, .01],
              [.01, 1]])

tf.reset_default_graph()
h_size = 100
a_dof = 2
c_dof = 6

# First agent
agent_1 = architecture.Agent(a_dof, c_dof, h_size, 'agent_1', batch, trace, tau)
agent_2 = architecture.Agent(a_dof, c_dof, h_size, 'agent_2', batch, trace, tau)

""" MODEL TRAINING"""

buffer = rl.PERBuffer(buffer_size, batch, trace, n_var)

# Launch the learning
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    agent_1.initialize_targets(sess)
    agent_2.initialize_targets(sess)
    
    # Iterate all the episodes
    for i in range(episodes):
        print("\nEPISODE: ", i)
        
        # Store cumulative reward per episode
        cum_r_list.append(cum_r)
        cum_r = 0
        
        # Store the experience from the episode
        episode_buffer = []
        
        # Instances of the environment
        gen_1 = dn.Generator(1.5, alpha=2)
        gen_2 = dn.Generator(1.5, alpha=1)
        
        network_node_1 = dn.NetworkNode(f_set_point=50, m=0.1, d=0.0160, t_g=30, r_d=0.1, idx=0)
        network_node_2 = dn.NetworkNode(f_set_point=50, m=0.15, d=0.0180, t_g=30, r_d=0.08, idx=1)

        network_node_1.set_load(1.5 + (- 0.25 + np.random.rand()/2))
        network_node_2.set_load(1.5 + (- 0.25 + np.random.rand() / 2))

        network_node_1.set_generation(1.5)
        network_node_2.set_generation(1.5)

        network = dn.Network([network_node_1, network_node_2], b)

        network_node_1.set_true_load(network.get_true_load(network_node_1.idx))
        network_node_2.set_true_load(network.get_true_load(network_node_2.idx))

        network_node_1.calculate_delta_f()
        network_node_2.calculate_delta_f()
        
        # Initial state for the LSTM
        st_1 = (np.zeros([1, h_size]), np.zeros([1, h_size]))
        st_2 = (np.zeros([1, h_size]), np.zeros([1, h_size]))

        p_ep = 0
        
        # Iterate all over the steps
        for j in range(steps):
            
            # Get the action from the actor and the internal state of the rnn
            curr_f_1 = network_node_1.get_delta_f()
            curr_f_2 = network_node_2.get_delta_f()

            curr_Z_1 = gen_1.get_z()
            curr_Z_2 = gen_2.get_z()
            
            # First agent
            a_1, new_st_1 = agent_1.a_actor_operation(sess, np.array([curr_f_1, curr_Z_1]).reshape(1, a_dof), st_1)
            a_1 = a_1[0, 0] + epsilon*np.random.normal(0.0, .01)
            
            # Second agent
            a_2, new_st_2 = agent_2.a_actor_operation(sess, np.array([curr_f_2, curr_Z_2]).reshape(1, a_dof), st_2)
            a_2 = a_2[0, 0] + epsilon*np.random.normal(0.0, .01)
            
            # Take the action, modify environment and get the reward
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

            new_f_1 = network_node_1.get_delta_f()
            new_f_2 = network_node_2.get_delta_f()

            r = rl.get_network_reward([network_node_1, network_node_2], e_f=.1)
            cum_r += r

            # PER
            a_1_n, _ = agent_1.a_actor_operation(sess, np.array([new_f_1,
                                                                 gen_1.get_z()]).reshape(1, a_dof), new_st_1)
            a_2_n, _ = agent_2.a_actor_operation(sess, np.array([new_f_2,
                                                                 gen_2.get_z()]).reshape(1, a_dof), new_st_2)

            p_1 = agent_1.importance(sess,
                                     np.array([curr_f_1, curr_Z_1, a_1, curr_f_2,
                                               curr_Z_2, a_2]).reshape(1, c_dof),
                                     np.array([new_f_1, gen_1.get_z(), a_1_n[0, 0],
                                               new_f_2, gen_2.get_z(), a_2_n[0, 0]]).reshape(1, c_dof),
                                     r, gamma, st_1, new_st_1)

            p_2 = agent_2.importance(sess,
                                     np.array([curr_f_2, curr_Z_2, a_2,
                                               curr_f_1, curr_Z_1, a_1]).reshape(1, c_dof),
                                     np.array([new_f_2, gen_2.get_z(), a_2_n[0, 0],
                                               new_f_1, gen_1.get_z(), a_1_n[0, 0]]).reshape(1, c_dof),
                                     r, gamma, st_2, new_st_2)

            p = p_1 + p_2
            p_ep += p
            p_tot += p_1 + p_2

            # Store the experience and print some data
            experience = np.array([curr_f_1, curr_f_2, curr_Z_1, curr_Z_2, gen_1.get_z(), gen_2.get_z(),
                                   new_f_1, new_f_2, a_1, a_2, r, p])
            episode_buffer.append(experience)

            print("Delta f1: {:+04.2f}   Delta f2: {:+04.2f}   Z1: {:05.2f}  Z2: {:05.2f}  Reward: {:04d}\
              Epsilon: {:05.4f}  p: {:06.1f}   a1: {:+04.2f}    a2: {:+04.2f}   Q1: {:+05.1f}  Q2: {:+05.1f}"
                  .format(curr_f_1, curr_f_2, curr_Z_1, curr_Z_2, r, epsilon, p, a_1, a_2,
                          sess.run(agent_1.critic.q, feed_dict={agent_1.critic.inp: np.array(
                              [curr_f_1,  curr_Z_1, a_1,  curr_f_2, curr_Z_2, a_2]).reshape(1, c_dof),
                                                                agent_1.critic.train_length: 1,
                                                                agent_1.critic.batch_size: 1,
                                                                agent_1.critic.state_in: (
                                                                    np.zeros([1, h_size]), np.zeros([1, h_size]))})[
                              0, 0],
                          sess.run(agent_2.critic.q, feed_dict={agent_2.critic.inp: np.array(
                              [curr_f_2, curr_Z_2, a_2, curr_f_1, curr_Z_1, a_1]).reshape(1, c_dof),
                                                                agent_2.critic.train_length: 1,
                                                                agent_2.critic.batch_size: 1,
                                                                agent_2.critic.state_in: (
                                                                    np.zeros([1, h_size]), np.zeros([1, h_size]))})[
                              0, 0]))
            
            # Update the model each 4 steps with a mini_batch of 32
            if ((j % 4) == 0) & (i > 0) & (i > 100):
                
                # Sample the mini_batch
                mini_batch = buffer.sample(p_tot)
                
                # Reset the recurrent layer's hidden state and get states
                s_1 = np.reshape(mini_batch[:, 0], [32, 1])
                s_2 = np.reshape(mini_batch[:, 1], [32, 1])
                Z1 = np.reshape(mini_batch[:, 2], [32, 1])
                Z2 = np.reshape(mini_batch[:, 3], [32, 1])
                Z1_p = np.reshape(mini_batch[:, 4], [32, 1])
                Z2_p = np.reshape(mini_batch[:, 5], [32, 1])
                s_p_1 = np.reshape(mini_batch[:, 6], [32, 1])
                s_p_2 = np.reshape(mini_batch[:, 7], [32, 1])
                a_1 = np.reshape(mini_batch[:, 8], [32, 1])
                a_2 = np.reshape(mini_batch[:, 9], [32, 1])
                rws = np.reshape(mini_batch[:, 10], [32, 1])

                # Predict the actions of both actors
                a_t_1 = agent_1.a_target_actor_training(sess, np.hstack((s_p_1, Z1_p)))
                a_t_2 = agent_2.a_target_actor_training(sess, np.hstack((s_p_2, Z2_p)))
                
                # Predict Q of the critics
                Q_target_1 = rws + gamma*agent_1.q_target_critic(sess,
                                                                 np.hstack((s_p_1, Z1_p, a_t_1, s_p_2, Z2_p, a_t_2)))
                Q_target_2 = rws + gamma*agent_2.q_target_critic(sess,
                                                                 np.hstack((s_p_2, Z2_p, a_t_2, s_p_1, Z1_p, a_t_1)))

                # Update the critic networks with the new Q's
                agent_1.update_critic(sess,
                                      np.hstack((s_1, Z1, a_1, s_2, Z2, a_2)), Q_target_1)
                agent_2.update_critic(sess,
                                      np.hstack((s_2, Z2, a_2, s_1, Z1, a_1)), Q_target_2)

                # Sample the new actions
                nw_a1 = agent_1.a_actor_training(sess, np.hstack((s_1, Z1)))
                nw_a2 = agent_2.a_actor_training(sess, np.hstack((s_2, Z2)))
                
                # Calculate the gradients
                grds_1 = agent_1.gradients_critic(sess,
                                                  np.hstack((s_1, Z1, nw_a1, s_2, Z2, nw_a2)))[0][:, 2].reshape(-1, 1)
                grds_2 = agent_2.gradients_critic(sess,
                                                  np.hstack((s_2, Z2, nw_a2, s_1, Z1, nw_a1)))[0][:, 2].reshape(-1, 1)

                # Update the actors
                agent_1.update_actor(sess, np.hstack((s_1, Z1)), grds_1)
                agent_2.update_actor(sess, np.hstack((s_2, Z2)), grds_2)

                # Update target network parameters
                agent_1.update_targets(sess)
                agent_2.update_targets(sess)
            
            # Update the state
            st_1 = new_st_1
            st_2 = new_st_2
            
            # Update epsilon
            epsilon = rl.get_new_epsilon(epsilon)
            
            # End episode if delta f is too large
            if (np.abs(network_node_1.get_delta_f()) > 10.0) | (np.abs(network_node_2.get_delta_f()) > 10.0):
                break
            
        # Append episode to the buffer
        if len(episode_buffer) >= 8:
            episode_buffer = np.array(episode_buffer)
            buffer.add(episode_buffer, p_ep)
            
    """ SAVE THE DATA"""

    saver = tf.train.Saver()
    saver.save(sess, "model/model_two_gens_network")
    with open("two_gens_network_reward.pickle", "wb") as handle:
        pck.dump(cum_r_list, handle, protocol=pck.HIGHEST_PROTOCOL)
