import numpy as np


class Generator:
    """ Implements each generation/load node

        Methods:
            modify_z : Increments/decrements the value of the control action.
            get_z: Getter of the control action z.
    """

    def __init__(self, z, alpha=1):
        """Constructor of Node class.

            Args:
                z (float): Initial control action of a given generator.
                alpha (float): Cost factor associated to a given generator.
        """

        self.z = z
        self.alpha = alpha
        
    def modify_z(self, delta_z):
        self.z += delta_z
        
        if self.z < 0.0:
            self.z = 0.0
            
        if self.z > 5.0:
            self.z = 5.0
        
    def get_z(self):
        return self.z

    def get_cost(self):
        return self.z*self.alpha


class Area:
    """ Implements each area frequency conditions in Secondary Control

        Methods:
            set_load: Setter of the area load.
            set_generation: Setter of the area generation.
            calculate_delta_f: Calculates the increase/decrease of the network frequency.
            calculate_p_g: Calculates new generation given the update of the control action.
            get_delta_f: Getter of the variation of the frequency.
            get_frequency: Getter of the frequency of the area.
            get_load: Getter of the load of the area.
            get_generation: Getter of the generation in the area.
    """

    def __init__(self, f_set_point, m, d, t_g, r_d):
        """Constructor of Area class.

            Args:
                f_set_point (float): Frequency set point of the network.
                m (float): inertia constant of the system.
                d (float): damping coefficient.
                t_g (float): time constant.
                r_d (float): droop.
        """

        self.f = f_set_point
        self.delta_f = 0
        self.m = m
        self.d = d
        self.t_g = t_g
        self.r_d = r_d

        self.p_l = 0
        self.p_g = 0
        self.delta_f = 0

        # This section corresponds to a Wiener generator, not always used
        self.A = np.array([[-0.002, 0.01],
                           [0, -0.5]])

        self.B = np.array([[0],
                           [-0.4]])

        self.dt = 1

        self.st = np.array([[0.5],
                            [0.0]])

        self.p_wiener = []
        
    def set_load(self, p_l):
        self.p_l = p_l
        
    def set_generation(self, p_g):
        self.p_g = p_g
    
    def calculate_delta_f(self, wiener=False):
        if wiener:
            self.st = self.st + self.dt * self.A @ self.st + self.dt * self.B * np.random.randn()
            self.p_wiener.append(self.st[0, 0])
            self.delta_f += (self.p_g - (self.p_l - self.st[0, 0]) - self.d * self.delta_f) / self.m

        else:
            self.delta_f += (self.p_g - self.p_l - self.d*self.delta_f)/self.m
        
    def calculate_p_g(self, z):
        self.p_g += (-self.p_g + z - (1/self.r_d)*self.delta_f)/self.t_g
        
    def get_delta_f(self):
        return self.delta_f
    
    def get_frequency(self):
        return self.f + self.delta_f
    
    def get_load(self):
        return self.p_l
    
    def get_generation(self):
        return self.p_g


class NetworkNode:

    def __init__(self, f_set_point, m, d, t_g, r_d, idx):
        """Constructor of Area class.

            Args:
                f_set_point (float): Frequency set point of the network.
                m (float): inertia constant of the system.
                d (float): damping coefficient.
                t_g (float): time constant.
                r_d (float): droop.
        """

        self.f = f_set_point
        self.delta_f = 0
        self.m = m
        self.d = d
        self.t_g = t_g
        self.r_d = r_d
        self.idx = idx

        self.p_l = 0
        self.p_i = 0
        self.p_g = 0
        self.nu = 0
        self.delta_f = 0

    def set_load(self, p_l):
        self.p_l = p_l

    def set_generation(self, p_g):
        self.p_g = p_g

    def set_true_load(self, p_i):
        self.p_i = p_i

    def calculate_delta_f(self):
        self.delta_f += (self.p_g - self.p_i - self.d * self.delta_f) / self.m

    def calculate_p_g(self, z):
        self.p_g += (-self.p_g + z - (1 / self.r_d) * self.delta_f) / self.t_g

    def calculate_nu(self):
        self.nu += self.delta_f

    def get_delta_f(self):
        return self.delta_f

    def get_frequency(self):
        return self.f + self.delta_f

    def get_load(self):
        return self.p_l

    def get_generation(self):
        return self.p_g

    def get_nu(self):
        return self.nu


class Network:

    def __init__(self, nodes, b, f_m):

        self.nodes = nodes
        self.b = b
        self.n_nodes = len(nodes)
        self.f_m = f_m

    def get_true_load(self, idx):

        load = self.nodes[idx].get_load()
        nu_diff = [self.b[idx, i]*(self.nodes[idx].get_nu() - self.nodes[i].get_nu()) for i in range(self.n_nodes)]
        nu_diff = [self.f_m if nu > self.f_m else -self.f_m if nu < -self.f_m else nu for nu in nu_diff]

        return load + sum(nu_diff)

