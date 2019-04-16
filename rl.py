import numpy as np


class ExperienceBuffer:
    """ Buffer to implement Experience Replay

        Methods:
            add: Add element to the buffer
            sample: Sample a group of elements from the buffer.
    """

    def __init__(self, buffer_size, batch_size, trace_length, n_var):
        """Constructor of Node class.

            Args:
                buffer_size (int): Size of the buffer.
                batch_size (int): Size of the batch to train
                trace_length (int): Size of the trace used to feed the LSTM
                n_var (int): Number of variables per experience
        """

        self.buffer = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.trace_length = trace_length
        self.n_var = n_var
    
    def add(self, experience):

        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1+len(self.buffer))-self.buffer_size] = []

        self.buffer.append(experience)
            
    def sample(self):

        index = np.random.choice(np.arange(len(self.buffer)), self.batch_size)
        sampled_episodes = [self.buffer[i] for i in index]
        sampled_traces = []

        for episode in sampled_episodes:
            point = np.random.randint(0, episode.shape[0]+1-self.trace_length)
            sampled_traces.append(episode[point:point+self.trace_length, :])

        return np.reshape(np.array(sampled_traces), [-1, self.n_var])


def get_new_epsilon(epsilon):
    """ Decay of epsilon over time.
        Args:
            epsilon (float): current epsilon.

        Returns:
            epsilon (float): new epsilon.
    """

    if epsilon < 0.5:
        return epsilon*0.99999

    return epsilon*0.999999

  
def get_reward(delta_f, z1, z2, e_f=.05, e_z=.2):
    """" Get reward from two agents.

        Args:
            delta_f (float): current deviance from network frequency set point.
            z1 (float): current control action of agent 1.
            z2 (float): current control action of agent 2.
            e_f (float): maximum error admitted in frequency dimension.
            e_z (float): maximum error admitted in cost dimension.

        Returns:
            epsilon (float): new epsilon.
    """
    if (z1 < .5) | (z2 < .5) | (z1 > 4.5) | (z2 > 4.5):
        return 0

    if (np.abs(delta_f) < e_f) & (np.abs(z1-(z2/2)) < e_z):
        return 200

    elif (np.abs(delta_f) < e_f) | (np.abs(z1 - (z2 / 2)) < e_z):
        return 100

    return 0


def get_network_reward(nodes, e_f=.1):

    return sum([100 if np.abs(node.get_delta_f()) < e_f else 0 for node in nodes])


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[data_idx]


class PERBuffer:

    def __init__(self, buffer_size, batch_size, train_length, n_vars):

        self.buffer = SumTree(buffer_size)
        self.n_vars = n_vars
        self.batch_size = batch_size
        self.train_length = train_length
        self.batch_length = self.train_length*self.batch_size

    def add(self, experience, p_ep):
        self.buffer.add(p_ep, experience)

        '''size = experience.shape[0]//self.batch_length
        chunks = experience[:self.batch_length*size, :].reshape(-1, self.batch_length, self.n_vars)

        for chunk in chunks:
            self.buffer.add(sum(chunk[:, -1]), chunk)'''

    def sample(self, p_tot):

        sampled_episodes = [self.buffer.get(np.random.randint(0, int(p_tot-1))) for i in range(self.batch_size)]
        sampled_traces = []

        for idx, p, episode in sampled_episodes:
            point = np.random.randint(0, episode.shape[0] + 1 - self.train_length)
            sampled_traces.append(episode[point:point + self.train_length, :])

        return np.array(sampled_traces).reshape(-1, self.n_vars)

        '''idx, p, data = self.buffer.get(np.random.randint(0, int(p_tot-1)))
        return data'''