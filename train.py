import dynamics as dn
import rl
import tensorflow as tf
import numpy as np
import architecture
import pickle as pck

""" MODEL DEFINITION"""

tf.reset_default_graph()
h_size = 100

# First agent
agent_1 = architecture.Agent(h_size, 'agent_1')
agent_2 = architecture.Agent(h_size, 'agent_2')

# Instantiate the model to initialize the variables
init = tf.global_variables_initializer()


""" MODEL TRAINING"""

# Parameters
gamma = 0.9
tau = 0.001
epsilon = 0.99
episodes = 10000
steps = 100
cum_r = 0
trace = 8
batch = 4
n_var = 9
cum_r_list = []

# Utils
agent_1.actor_target.create_op_holder(agent_1.actor.network_params, tau)
agent_1.critic_target.create_op_holder(agent_1.critic.network_params, tau)

agent_2.actor_target.create_op_holder(agent_2.actor.network_params, tau)
agent_2.critic_target.create_op_holder(agent_2.critic.network_params, tau)

buffer = rl.ExperienceBuffer(1000)

# Launch the learning
with tf.Session() as session:
    session.run(init)
    
    # Iterate all the episodes
    for i in range(episodes):
        print("\nEPISODE: ", i)
        
        # Store cumulative reward per episode
        cum_r_list.append(cum_r)
        cum_r = 0
        
        # Store the experience from the episode
        episode_buffer = []
        
        # Instances of the environment
        generator_1 = dn.Node(1.5, alpha=2)
        generator_2 = dn.Node(1.5, alpha=1)
        
        area = dn.Area(f_set_point=50, m=0.1, d=0.0160, t_g=30, r_d=0.1)
        area.set_load(3.0 + (- 0.25 + np.random.rand()/2))
        area.set_generation(3.0)
        area.calculate_delta_f()
        
        # Initial state for the LSTM
        state_1 = (np.zeros([1, h_size]), np.zeros([1, h_size]))
        state_2 = (np.zeros([1, h_size]), np.zeros([1, h_size]))
        
        # Iterate all over the steps
        for j in range(steps):
            
            # Get the action from the actor and the internal state of the rnn
            current_f = area.get_delta_f()
            current_Z_1 = generator_1.get_z()
            current_Z_2 = generator_2.get_z()
            
            # First agent
            a_1, new_state_1 = session.run([agent_1.actor.a, agent_1.actor.rnn_state], 
                                           feed_dict={agent_1.actor.f: np.array(current_f).reshape(1, 1),
                                                      agent_1.actor.p: np.array(current_Z_1).reshape(1, 1),
                                                      agent_1.actor.state_in: state_1, agent_1.actor.batch_size: 1, 
                                                      agent_1.actor.train_length: 1})
            a_1 = a_1[0, 0] + epsilon*np.random.normal(0.0, 0.2)
            
            # Second agent
            a_2, new_state_2 = session.run([agent_2.actor.a, agent_2.actor.rnn_state], 
                                           feed_dict={agent_2.actor.f: np.array(current_f).reshape(1, 1),
                                                      agent_2.actor.p: np.array(current_Z_2).reshape(1, 1),
                                                      agent_2.actor.state_in: state_2, agent_2.actor.batch_size: 1,
                                                      agent_2.actor.train_length: 1})
            a_2 = a_2[0, 0] + epsilon*np.random.normal(0.0, 0.2)
            
            # Take the action, modify environment and get the reward
            generator_1.modify_z(a_1)
            generator_2.modify_z(a_2)
            Z = generator_1.get_z() + generator_2.get_z()
            area.calculate_p_g(Z)
            area.calculate_delta_f()
            new_f = area.get_delta_f()
            r = rl.get_reward(new_f, generator_1.get_z(), generator_2.get_z(), e_f=.05, e_z=.1, combined=True)
            cum_r += r

            # Store the experience and print some data
            experience = np.array([current_f, current_Z_1, current_Z_2, generator_1.get_z(), generator_2.get_z(),
                                   new_f, a_1, a_2, r])
            episode_buffer.append(experience)
            print("Delta f: ", round(current_f, 2), " P1: ", current_Z_1,
                  " P2: ", current_Z_2, " Reward: ", r)
            
            # Update the model each 4 steps with a mini_batch of 32
            if ((j % 4) == 0) & (i > 0) & (len(buffer.buffer) > 0):
                
                # Sample the mini_batch
                mini_batch = buffer.sample(batch, trace, n_var)
                
                # Reset the recurrent layer's hidden state and get states
                state_train = (np.zeros([batch, h_size]), np.zeros([batch, h_size]))
                s = np.reshape(mini_batch[:, 0], [32, 1])
                Z1 = np.reshape(mini_batch[:, 1], [32, 1])
                Z2 = np.reshape(mini_batch[:, 2], [32, 1])
                Z1_prime = np.reshape(mini_batch[:, 3], [32, 1])
                Z2_prime = np.reshape(mini_batch[:, 4], [32, 1])
                s_prime = np.reshape(mini_batch[:, 5], [32, 1])
                actions_1 = np.reshape(mini_batch[:, 6], [32, 1])
                actions_2 = np.reshape(mini_batch[:, 7], [32, 1])
                rewards = np.reshape(mini_batch[:, 8], [32, 1])

                # Predict the actions of both actors
                a_target_1 = session.run(agent_1.actor_target.a, feed_dict={agent_1.actor_target.f: s_prime, 
                                                                            agent_1.actor_target.p: Z1_prime,
                                                                            agent_1.actor_target.state_in: state_train,
                                                                            agent_1.actor_target.batch_size: batch,
                                                                            agent_1.actor_target.train_length: trace})
                a_target_2 = session.run(agent_2.actor_target.a, feed_dict={agent_2.actor_target.f: s_prime, 
                                                                            agent_2.actor_target.p: Z2_prime,
                                                                            agent_2.actor_target.state_in: state_train,
                                                                            agent_2.actor_target.batch_size: batch,
                                                                            agent_2.actor_target.train_length: trace})
                
                # Predict Q of the critics
                Q_t_1 = session.run(agent_1.critic_target.q, feed_dict={agent_1.critic_target.f: s_prime,
                                                                        agent_1.critic_target.p: Z1_prime,
                                                                        agent_1.critic_target.a: a_target_1,
                                                                        agent_1.critic_target.p_o: Z2_prime,
                                                                        agent_1.critic_target.a_o: a_target_2,
                                                                        agent_1.critic_target.train_length: trace,
                                                                        agent_1.critic_target.batch_size: batch,
                                                                        agent_1.critic_target.state_in: state_train})
                Q_t_2 = session.run(agent_2.critic_target.q, feed_dict={agent_2.critic_target.f: s_prime,
                                                                        agent_2.critic_target.p: Z2_prime,
                                                                        agent_2.critic_target.a: a_target_2,
                                                                        agent_2.critic_target.p_o: Z1_prime,
                                                                        agent_2.critic_target.a_o: a_target_1,
                                                                        agent_2.critic_target.train_length: trace,
                                                                        agent_2.critic_target.batch_size: batch,
                                                                        agent_2.critic_target.state_in: state_train})
                Q_target_1 = rewards + gamma*Q_t_1
                Q_target_2 = rewards + gamma*Q_t_2

                # Update the critic networks with the new Q's
                session.run(agent_1.critic.upd, feed_dict={agent_1.critic.f: s,
                                                           agent_1.critic.a: actions_1,
                                                           agent_1.critic.p: Z1,
                                                           agent_1.critic.a_o: actions_2,
                                                           agent_1.critic.target_q: Q_target_1,
                                                           agent_1.critic.p_o: Z2,
                                                           agent_1.critic.train_length: trace,
                                                           agent_1.critic.batch_size: batch,
                                                           agent_1.critic.state_in: state_train})
                session.run(agent_2.critic.upd, feed_dict={agent_2.critic.f: s, 
                                                           agent_2.critic.a: actions_2,
                                                           agent_2.critic.p: Z2,
                                                           agent_2.critic.a_o: actions_1,
                                                           agent_2.critic.target_q: Q_target_2,
                                                           agent_2.critic.p_o: Z1,
                                                           agent_2.critic.train_length: trace,
                                                           agent_2.critic.batch_size: batch,
                                                           agent_2.critic.state_in: state_train})
    
                # Sample the new actions
                new_a_1 = session.run(agent_1.actor.a, feed_dict={agent_1.actor.f: s, 
                                                                  agent_1.actor.p: Z1,
                                                                  agent_1.actor.state_in: state_train,
                                                                  agent_1.actor.batch_size: batch,
                                                                  agent_1.actor.train_length: trace})
                new_a_2 = session.run(agent_2.actor.a, feed_dict={agent_2.actor.f: s, 
                                                                  agent_2.actor.p: Z2,
                                                                  agent_2.actor.state_in: state_train,
                                                                  agent_2.actor.batch_size: batch,
                                                                  agent_2.actor.train_length: trace})
                
                # Calculate the gradients
                grads_1 = session.run(agent_1.critic.critic_gradients, feed_dict={agent_1.critic.f: s,
                                                                                  agent_1.critic.a: new_a_1,
                                                                                  agent_1.critic.p: Z1,
                                                                                  agent_1.critic.a_o: new_a_2,
                                                                                  agent_1.critic.train_length: trace,
                                                                                  agent_1.critic.batch_size: batch,
                                                                                  agent_1.critic.state_in: state_train,
                                                                                  agent_1.critic.p_o: Z2})
                grads_2 = session.run(agent_2.critic.critic_gradients, feed_dict={agent_2.critic.f: s,
                                                                                  agent_2.critic.a: new_a_2,
                                                                                  agent_2.critic.p: Z2,
                                                                                  agent_2.critic.a_o: new_a_1,
                                                                                  agent_2.critic.train_length: trace,
                                                                                  agent_2.critic.batch_size: batch,
                                                                                  agent_2.critic.state_in: state_train,
                                                                                  agent_2.critic.p_o: Z1})
                gradients_1 = grads_1[0]
                gradients_2 = grads_2[0]
                
                # Update the actors
                session.run(agent_1.actor.upd, feed_dict={agent_1.actor.f: s, 
                                                          agent_1.actor.p: Z1,
                                                          agent_1.actor.state_in: state_train,
                                                          agent_1.actor.critic_gradient: gradients_1,
                                                          agent_1.actor.batch_size: batch,
                                                          agent_1.actor.train_length: trace})
                session.run(agent_2.actor.upd, feed_dict={agent_2.actor.f: s, 
                                                          agent_2.actor.p: Z2,
                                                          agent_2.actor.state_in: state_train,
                                                          agent_2.actor.critic_gradient: gradients_2,
                                                          agent_2.actor.batch_size: batch,
                                                          agent_2.actor.train_length: trace})

                # Update target network parameters
                session.run(agent_1.actor_target.update_parameters)
                session.run(agent_1.critic_target.update_parameters)
                session.run(agent_2.actor_target.update_parameters)
                session.run(agent_2.critic_target.update_parameters)
            
            # Update the state
            state_1 = new_state_1
            state_2 = new_state_2
            
            # Update epsilon
            epsilon = rl.get_new_epsilon(epsilon)
            
            # End episode if delta f is too large
            if np.abs(area.get_delta_f()) > 50:
                break
            
        # Append episode to the buffer
        if len(episode_buffer) >= 8:
            episode_buffer = np.array(episode_buffer)
            if np.count_nonzero(episode_buffer[:, -1]) > 0:
                buffer.add(episode_buffer)
            
    """ SAVE THE DATA"""

    saver = tf.train.Saver()
    saver.save(session, "model/model")
    with open("reward.pickle", "wb") as handle:
        pck.dump(cum_r_list, handle, protocol=pck.HIGHEST_PROTOCOL)
