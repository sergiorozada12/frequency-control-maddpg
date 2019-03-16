import tensorflow as tf


class Actor:

    def __init__(self, h_size, cell, name, num_variables):
        """Constructor of Actor class.

            Args:
                h_size (int): Size of the LSTM output.
                cell: LSTM cell for the network.
                name (str): name of the context.
                num_variables (int): number of parameters of the network.
        """

        # Input
        self.f = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.p = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.inp = tf.concat([self.f, self.p], axis=1)
        self.initializer = tf.contrib.layers.xavier_initializer()
        
        # LSTM
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
        self.train_length = tf.placeholder(dtype=tf.int32)
        self.rnn_inp = tf.reshape(self.inp, [self.batch_size, self.train_length, 2])
        
        self.state_in = cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=self.rnn_inp, cell=cell,
                                                     dtype=tf.float32, initial_state=self.state_in, scope=name+'_rnn')
        self.rnn = tf.reshape(self.rnn, shape=[-1, h_size])
        
        # MLP
        self.b1 = tf.Variable(self.initializer([1, 1000]))
        self.W1 = tf.Variable(self.initializer([h_size, 1000]))
        self.h1 = tf.nn.relu(tf.matmul(self.rnn, self.W1)+self.b1)

        self.b2 = tf.Variable(self.initializer([1, 100]))
        self.W2 = tf.Variable(self.initializer([1000, 100]))
        self.h2 = tf.nn.relu(tf.matmul(self.h1, self.W2)+self.b2)

        self.b3 = tf.Variable(self.initializer([1, 50]))
        self.W3 = tf.Variable(self.initializer([100, 50]))
        self.h3 = tf.nn.relu(tf.matmul(self.h2, self.W3)+self.b3)

        self.b4 = tf.Variable(self.initializer([1, 1]))
        self.W4 = tf.Variable(self.initializer([50, 1]))
        self.a_unscaled = tf.nn.tanh(tf.matmul(self.h3, self.W4)+self.b4)
        self.a = tf.multiply(self.a_unscaled, 0.1)

        # Gradients
        self.network_params = tf.trainable_variables()[num_variables:]
        self.critic_gradient = tf.placeholder(tf.float32, [None, 1])
        self.unnormalized_actor_gradients = tf.gradients(self.a, self.network_params, - self.critic_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, 32), self.unnormalized_actor_gradients))
        
        # Optimization
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.upd = self.optimizer.apply_gradients(zip(self.actor_gradients, self.network_params))

        self.update_parameters = []

    def create_op_holder(self, params, tau):
        """ Use target network op holder if needed"""
        self.update_parameters = [self.network_params[i].assign(tf.multiply(params[i], tau) +
                                                                tf.multiply(self.network_params[i], 1. - tau))
                                  for i in range(len(self.network_params))]


class Critic:

    def __init__(self, h_size, cell, name, num_variables):
        """Constructor of Critic class.

            Args:
                h_size (int): Size of the LSTM output.
                cell: LSTM cell for the network.
                name (str): name of the context.
                num_variables (int): number of parameters of the network.
        """

        # Input
        self.f = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.p = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.p_o = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.a = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.a_o = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.inp = tf.concat([self.f, self.p, self.p_o, self.a, self.a_o], axis=1)
        self.initializer = tf.contrib.layers.xavier_initializer()
        
        # LSTM
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
        self.train_length = tf.placeholder(dtype=tf.int32)
        self.rnn_inp = tf.reshape(self.inp, [self.batch_size, self.train_length, 5])
        
        self.state_in = cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=self.rnn_inp, cell=cell,
                                                     dtype=tf.float32, initial_state=self.state_in, scope=name+'_rnn')
        self.rnn = tf.reshape(self.rnn, shape=[-1, h_size])
        
        # MLP
        self.b1 = tf.Variable(self.initializer([1, 1000]))
        self.W1 = tf.Variable(self.initializer([h_size, 1000]))
        self.h1 = tf.nn.relu(tf.matmul(self.rnn, self.W1)+self.b1)

        self.b2 = tf.Variable(self.initializer([1, 100]))
        self.W2 = tf.Variable(self.initializer([1000, 100]))
        self.h2 = tf.nn.relu(tf.matmul(self.h1, self.W2)+self.b2)

        self.b3 = tf.Variable(self.initializer([1, 50]))
        self.W3 = tf.Variable(self.initializer([100, 50]))
        self.h3 = tf.nn.relu(tf.matmul(self.h2, self.W3)+self.b3)

        self.b4 = tf.Variable(self.initializer([1, 1]))
        self.W4 = tf.Variable(self.initializer([50, 1]))
        self.q = tf.matmul(self.h3, self.W4)+self.b4
        
        # Gradients
        self.network_params = tf.trainable_variables()[num_variables:]
        self.target_q = tf.placeholder(tf.float32, [None, 1])
        
        # Optimization
        self.loss = tf.reduce_mean(tf.square(self.target_q-self.q))
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.upd = self.optimizer.minimize(self.loss)
        
        # Gradients
        self.critic_gradients = tf.gradients(self.q, self.a)

        self.update_parameters = []

    def create_op_holder(self, params, tau):
        """ Use target network op holder if needed"""
        self.update_parameters = [self.network_params[i].assign(tf.multiply(params[i], tau) +
                                                                tf.multiply(self.network_params[i], 1. - tau))
                                  for i in range(len(self.network_params))]
