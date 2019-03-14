import numpy as np
import pickle as pck


class ExperienceBuffer:
    """ Buffer to implement Experience Replay

        Methods:
            add: Add element to the buffer
            sample: Sample a group of elements from the buffer.
    """

    def __init__(self, buffer_size):
        """Constructor of Node class.

            Args:
                buffer_size (int): Size of the buffer.
        """

        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self, experience):

        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1+len(self.buffer))-self.buffer_size] = []

        self.buffer.append(experience)
            
    def sample(self, batch_size, trace_length, n_var):

        index = np.random.choice(np.arange(len(self.buffer)), batch_size)
        sampled_episodes = [self.buffer[i] for i in index]
        sampled_traces = []

        for episode in sampled_episodes:
            point = np.random.randint(0, episode.shape[0]+1-trace_length)
            sampled_traces.append(episode[point:point+trace_length, :])

        return np.reshape(np.array(sampled_traces), [-1, n_var])


def get_new_epsilon(epsilon, decay_rate=.9):
    """ Decay of epsilon over time.
        Args:
            epsilon (float): current epsilon.
            decay_rate (float): rate of decayment.

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
    
    r = 0
    
    if np.abs(delta_f) < .05:
        r = 100
        
    if np.abs(z1-(z2/2)) < .2:
        r += 100
    
    return r





def setContinuousPower(action,node):
    """ Perform agent action"""
    node.modifyPower(action)       
    return node

def getSumZ(nodes):
    Z = 0
    for node in nodes:
        Z += node.getZ()
    return Z

def endEpisode(delta_f):
    """ End the episode if frequency is 10 Hertz far away from setpoint"""
    if np.abs(delta_f) > 50:
        return True
    return False

def saveData(name,data):
    """ Save information about the trained model"""
    with open("data/"+name, "wb") as handle:
        pck.dump(data, handle, protocol=pck.HIGHEST_PROTOCOL)
    
def readData(name):
    """ Load the requested model"""
    with open("../data/"+name, "rb") as handle:
        return pck.load(handle)




    
def updateTargetGraph(tfVars,tau):
    """ Update target graph towards main graph"""
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    """ Commit the update"""
    for op in op_holder:
        sess.run(op)
        
def discountReward(r,gamma):
    """ Discount the set of rewards and normalize them"""
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r) 
    return discounted_r
