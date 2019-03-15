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
lstm_actor_1 = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
lstm_critic_1 = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
actor_1 = architecture.Actor(h_size, lstm_actor_1, 'actor_1', 0)
critic_1 = architecture.Critic(h_size, lstm_critic_1, 'critic_1', len(tf.trainable_variables()))

lstm_actor_t_1 = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
lstm_critic_t_1 = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
actor_t_1 = architecture.Actor(h_size, lstm_actor_t_1, 'actor_t_1', len(tf.trainable_variables()))
critic_t_1 = architecture.Critic(h_size, lstm_critic_t_1, 'critic_t_1', len(tf.trainable_variables()))

# Second agent
lstm_actor_2 = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
lstm_critic_2 = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
actor_2 = architecture.Actor(h_size, lstm_actor_2, 'actor_2', len(tf.trainable_variables()))
critic_2 = architecture.Critic(h_size, lstm_critic_2, 'critic_2', len(tf.trainable_variables()))

lstm_actor_t_2 = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
lstm_critic_t_2 = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
actor_t_2 = architecture.Actor(h_size, lstm_actor_t_2, 'actor_t_2', len(tf.trainable_variables()))
critic_t_2 = architecture.Critic(h_size, lstm_critic_t_2, 'critic_t_2', len(tf.trainable_variables()))

# Instantiate the model to initialize the variables
init = tf.global_variables_initializer()


""" MODEL TRAINING"""

# Parameters
gamma = 0.9
tau = 0.001
epsilon = 0.99
episodes = 50000
steps = 100
cum_r = 0
trace = 8
batch = 4
n_var = 9
cum_r_list = []

# Utils
actor_t_1.create_op_holder(actor_1.network_params, tau)
critic_t_1.create_op_holder(critic_1.network_params, tau)

actor_t_2.create_op_holder(actor_2.network_params, tau)
critic_t_2.create_op_holder(critic_2.network_params, tau)

buffer = rl.ExperienceBuffer(10000)

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
            a_1, new_state_1 = session.run([actor_1.a, actor_1.rnn_state], 
                                           feed_dict={actor_1.f: np.array(current_f).reshape(1, 1),
                                                      actor_1.p: np.array(current_Z_1).reshape(1, 1),
                                                      actor_1.state_in: state_1, actor_1.batch_size: 1, 
                                                      actor_1.train_length: 1})
            a_1 = a_1[0, 0] + epsilon*np.random.normal(0.0, 0.2)
            
            # Second agent
            a_2, new_state_2 = session.run([actor_2.a, actor_2.rnn_state], 
                                           feed_dict={actor_2.f: np.array(current_f).reshape(1, 1),
                                                      actor_2.p: np.array(current_Z_2).reshape(1, 1),
                                                      actor_2.state_in: state_2, actor_2.batch_size: 1,
                                                      actor_2.train_length: 1})
            a_2 = a_2[0, 0] + epsilon*np.random.normal(0.0, 0.2)
            
            # Take the action, modify environment and get the reward
            generator_1.modify_z(a_1)
            generator_2.modify_z(a_2)
            Z = generator_1.get_z() + generator_2.get_z()
            area.calculate_p_g(Z)
            area.calculate_delta_f()
            new_f = area.get_delta_f()
            r = rl.get_reward(new_f, generator_1.get_z(), generator_2.get_z(), e_f=.05, e_z=.2)
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
                a_target_1 = session.run(actor_t_1.a, feed_dict={actor_t_1.f: s_prime, 
                                                                 actor_t_1.p: Z1_prime, 
                                                                 actor_t_1.state_in: state_train, 
                                                                 actor_t_1.batch_size: batch,
                                                                 actor_t_1.train_length: trace})
                a_target_2 = session.run(actor_t_2.a, feed_dict={actor_t_2.f: s_prime, 
                                                                 actor_t_2.p: Z2_prime, 
                                                                 actor_t_2.state_in: state_train, 
                                                                 actor_t_2.batch_size: batch,
                                                                 actor_t_2.train_length: trace})
                
                # Predict Q of the critics
                Q_target_1 = session.run(critic_t_1.q, feed_dict={critic_t_1.f: s_prime, 
                                                                  critic_t_1.p: Z1_prime,
                                                                  critic_t_1.a: a_target_1,
                                                                  critic_t_1.p_o: Z2_prime,
                                                                  critic_t_1.a_o: a_target_2, 
                                                                  critic_t_1.train_length: trace,
                                                                  critic_t_1.batch_size: batch, 
                                                                  critic_t_1.state_in: state_train})   
                Q_target_2 = session.run(critic_t_2.q, feed_dict={critic_t_2.f: s_prime, 
                                                                  critic_t_2.p: Z2_prime, 
                                                                  critic_t_2.a: a_target_2,
                                                                  critic_t_2.p_o: Z1_prime,
                                                                  critic_t_2.a_o: a_target_1,
                                                                  critic_t_2.train_length: trace,
                                                                  critic_t_2.batch_size: batch, 
                                                                  critic_t_2.state_in: state_train})
                Q_target_1 = rewards + gamma*Q_target_1
                Q_target_2 = rewards + gamma*Q_target_2

                # Update the critic networks with the new Q's
                session.run(critic_1.upd, feed_dict={critic_1.f: s,
                                                     critic_1.a: actions_1, 
                                                     critic_1.p: Z1, 
                                                     critic_1.a_o: actions_2,
                                                     critic_1.target_q: Q_target_1,
                                                     critic_1.p_o: Z2,
                                                     critic_1.train_length: trace,
                                                     critic_1.batch_size: batch, 
                                                     critic_1.state_in: state_train})  
                session.run(critic_2.upd, feed_dict={critic_2.f: s, 
                                                     critic_2.a: actions_2, 
                                                     critic_2.p: Z2, 
                                                     critic_2.a_o: actions_1, 
                                                     critic_2.target_q: Q_target_2,
                                                     critic_2.p_o: Z1,
                                                     critic_2.train_length: trace, 
                                                     critic_2.batch_size: batch,
                                                     critic_2.state_in: state_train}) 
    
                # Sample the new actions
                new_a_1 = session.run(actor_1.a, feed_dict={actor_1.f: s, 
                                                            actor_1.p: Z1,
                                                            actor_1.state_in: state_train, 
                                                            actor_1.batch_size: batch, 
                                                            actor_1.train_length: trace})
                new_a_2 = session.run(actor_2.a, feed_dict={actor_2.f: s, 
                                                            actor_2.p: Z2,
                                                            actor_2.state_in: state_train, 
                                                            actor_2.batch_size: batch, 
                                                            actor_2.train_length: trace})
                
                # Calculate the gradients
                gradients_1 = session.run(critic_1.critic_gradients, feed_dict={critic_1.f: s,
                                                                                critic_1.a: new_a_1,
                                                                                critic_1.p: Z1, 
                                                                                critic_1.a_o: new_a_2,
                                                                                critic_1.train_length: trace,
                                                                                critic_1.batch_size: batch,
                                                                                critic_1.state_in: state_train,
                                                                                critic_1.p_o: Z2})
                gradients_2 = session.run(critic_2.critic_gradients, feed_dict={critic_2.f: s,
                                                                                critic_2.a: new_a_2,
                                                                                critic_2.p: Z2, 
                                                                                critic_2.a_o: new_a_1,
                                                                                critic_2.train_length: trace,
                                                                                critic_2.batch_size: batch,
                                                                                critic_2.state_in: state_train,
                                                                                critic_2.p_o: Z1})
                gradients_1 = gradients_1[0]
                gradients_2 = gradients_2[0]
                
                # Update the actors
                session.run(actor_1.upd, feed_dict={actor_1.f: s, 
                                                    actor_1.p: Z1,
                                                    actor_1.state_in: state_train,
                                                    actor_1.critic_gradient: gradients_1, 
                                                    actor_1.batch_size: batch, 
                                                    actor_1.train_length: trace})
                session.run(actor_2.upd, feed_dict={actor_2.f: s, 
                                                    actor_2.p: Z2,
                                                    actor_2.state_in: state_train, 
                                                    actor_2.critic_gradient: gradients_2,
                                                    actor_2.batch_size: batch, 
                                                    actor_2.train_length: trace})

                # Update target network parameters
                session.run(actor_t_1.update_network_params)
                session.run(critic_t_1.update_network_params)
                session.run(actor_t_2.update_network_params)
                session.run(critic_t_2.update_network_params)
            
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
            buffer.add(np.array(episode_buffer))
            
    """ SAVE THE DATA"""
    saver = tf.train.Saver()
    saver.save(session, "model")
    with open("reward.pickle", "wb") as handle:
        pck.dump(cum_r_list, handle, protocol=pck.HIGHEST_PROTOCOL)
