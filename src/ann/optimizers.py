import numpy as np


class SGD:
    def __init__(self,lr=0.01,weight_decay=0.0):
        self.lr = lr
        self.weight_decay = weight_decay
    def update(self,layer):
        # get gradients
        grad_W = layer.grad_W
        grad_b = layer.grad_b
        # update weights and biases
        layer.W = layer.W - self.lr * grad_W - self.lr * self.weight_decay * layer.W
        layer.b = layer.b - self.lr*grad_b
    def step(self):
        # Used only in case of ADAM and NADAM
        pass
    def get_name(self):
        return "sgd"


class NAG:
    def __init__(self,learning_rate=0.01,beta=0.9,weight_decay=0.0):
        self.lr=learning_rate
        self.beta=beta
        self.weight_decay=weight_decay
        self.velocities_W = {}
        self.velocities_b= {}
    def update(self,layer):
        layer_id = id(layer)
        if layer_id not in self.velocities_W:
            self.velocities_W[layer_id]=np.zeros_like(layer.W)
            self.velocities_b[layer_id]=np.zeros_like(layer.b)
        # update velocity (gradient computed at lookhead position)
        self.velocities_W[layer_id] = (self.beta*self.velocities_W[layer_id]+ self.lr*layer.grad_W)
        self.velocities_b[layer_id] = (self.beta*self.velocities_b[layer_id] + self.lr*layer.grad_b)
        # update weights
        layer.W = layer.W - self.velocities_W[layer_id]-self.lr*self.weight_decay*layer.W
        layer.b = layer.b - self.velocities_b[layer_id]

    def step(self):
        pass

    def get_name(self):
        return "NAG"


class Momentum:
    def __init__(self,learning_rate=0.01,beta=0.9,weight_decay=0.0):
        self.beta=beta
        self.lr = learning_rate
        self.weight_decay=weight_decay
        self.velocities_W = {}
        self.velocities_b ={}
    def update(self,layer):
        grad_W = layer.grad_W
        grad_b = layer.grad_b
        layer_id = id(layer)

        if layer_id not in self.velocities_W:
            self.velocities_W[layer_id]=np.zeros_like(layer.W)
            self.velocities_b[layer_id]=np.zeros_like(layer.b)
        self.velocities_W[layer_id]=self.beta*self.velocities_W[layer_id] + (1-self.beta)*grad_W
        self.velocities_b[layer_id] = self.beta*self.velocities_b[layer_id] + (1-self.beta)*grad_b

        layer.W = layer.W - self.lr * self.velocities_W[layer_id] - self.lr * self.weight_decay * layer.W
        layer.b = layer.b - self.lr*self.velocities_b[layer_id]
    def get_name(self):
        return "momentum"
    def step(self):
        pass


class RMSprop:
    def __init__(self,learning_rate=0.01,beta=0.9,epsilon=1e-8,weight_decay=0.0):
        self.lr=learning_rate
        self.beta=beta
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.ms_W = {}  # mean square gradients for weights
        self.ms_b = {}  # mean square gradients for biases

    def update(self, layer):
        layer_id = id(layer)
        
        if layer_id not in self.ms_W:
            self.ms_W[layer_id] = np.zeros_like(layer.W)
            self.ms_b[layer_id] = np.zeros_like(layer.b)
        
        # Update mean squared gradients
        self.ms_W[layer_id] = self.beta * self.ms_W[layer_id] + (1 - self.beta) * (layer.grad_W ** 2)
        self.ms_b[layer_id] = self.beta * self.ms_b[layer_id] + (1 - self.beta) * (layer.grad_b ** 2)
        
        # Update weights (adaptive learning rate)
        layer.W = layer.W - self.lr * (layer.grad_W / (np.sqrt(self.ms_W[layer_id]) + self.epsilon)) - self.lr * self.weight_decay * layer.W
        layer.b = layer.b - self.lr * (layer.grad_b / (np.sqrt(self.ms_b[layer_id]) + self.epsilon))
    def get_name(self):
        return "rmsprop"
    def step(self):
        pass
class Adam:

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, 
                epsilon=1e-8, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay  
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Time step
        self.m_W = {}  # First moment (mean of gradients)
        self.v_W = {}  # Second moment (variance of gradients)
        self.m_b = {}  # First moment for biases
        self.v_b = {}  # Second moment for biases

    def update(self,layer):
        layer_id = id(layer)

    
        # Initialize moments if first time seeing layer
        if layer_id not in self.m_W:
            self.m_W[layer_id] = np.zeros_like(layer.W)
            self.v_W[layer_id] = np.zeros_like(layer.W)
            self.m_b[layer_id] = np.zeros_like(layer.b)
            self.v_b[layer_id] = np.zeros_like(layer.b)
        #  Update first moment estimate (Momentum part)
        # m = beta1 * m + (1 - beta1) * grad
        self.m_W[layer_id] = (
            self.beta1 * self.m_W[layer_id]
            + (1 - self.beta1) * layer.grad_W
        )
        self.m_b[layer_id] = (
            self.beta1 * self.m_b[layer_id]
            + (1 - self.beta1) * layer.grad_b
        )
        
        # Update second moment estimate (RMSProp part)
        # v = beta2 * v + (1 - beta2) * grad^2
        self.v_W[layer_id] = (self.beta2 * self.v_W[layer_id]+(1 - self.beta2) * (layer.grad_W ** 2))

        self.v_b[layer_id] = (self.beta2 * self.v_b[layer_id]+(1 - self.beta2) * (layer.grad_b ** 2))

        
        #  Bias correction
        # (Because m and v start at zero)
        m_W_hat = self.m_W[layer_id] / (1 - self.beta1 ** self.t)
        v_W_hat = self.v_W[layer_id] / (1 - self.beta2 ** self.t)

        m_b_hat = self.m_b[layer_id] / (1 - self.beta1 ** self.t)
        v_b_hat = self.v_b[layer_id] / (1 - self.beta2 ** self.t)

        
        #  Update parameters
        # W = W - lr * (m_hat / (sqrt(v_hat) + epsilon))
        # Weight update (with optional L2 regularization)
        layer.W -= self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
        layer.W -= self.learning_rate * self.weight_decay * layer.W

        # Bias update (no weight decay)
        layer.b -= self.learning_rate * (m_b_hat / (np.sqrt(v_b_hat) + self.epsilon))
    def step(self):
        """Increment the global time-step counter once per training batch."""
        self.t += 1

    def get_name(self):
        return "adam"

class Nadam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, 
                epsilon=1e-8, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay  
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Time step
        self.m_W = {}  # First moment (mean of gradients)
        self.v_W = {}  # Second moment (variance of gradients)
        self.m_b = {}  # First moment for biases
        self.v_b = {}  # Second moment for biases
    def update(self,layer):
        layer_id = id(layer)
        
        if layer_id not in self.m_W:
            self.m_W[layer_id] = np.zeros_like(layer.W)
            self.v_W[layer_id] = np.zeros_like(layer.W)
            self.m_b[layer_id] = np.zeros_like(layer.b)
            self.v_b[layer_id] = np.zeros_like(layer.b)
        
        # Update biased first moment
        self.m_W[layer_id] = self.beta1 * self.m_W[layer_id] + (1 - self.beta1) * layer.grad_W
        self.m_b[layer_id] = self.beta1 * self.m_b[layer_id] + (1 - self.beta1) * layer.grad_b
        
        # Update biased second moment
        self.v_W[layer_id] = self.beta2 * self.v_W[layer_id] + (1 - self.beta2) * (layer.grad_W ** 2)
        self.v_b[layer_id] = self.beta2 * self.v_b[layer_id] + (1 - self.beta2) * (layer.grad_b ** 2)
        
        # Bias correction
        m_W_hat = self.m_W[layer_id] / (1 - self.beta1 ** self.t)
        m_b_hat = self.m_b[layer_id] / (1 - self.beta1 ** self.t)
        v_W_hat = self.v_W[layer_id] / (1 - self.beta2 ** self.t)
        v_b_hat = self.v_b[layer_id] / (1 - self.beta2 ** self.t)
        
        # Nadam: Use Nesterov momentum (look-ahead)
        # m_hat_nesterov = β₁*m_hat + (1-β₁)*gradient
        m_W_nesterov = self.beta1 * m_W_hat + (1 - self.beta1) * layer.grad_W
        m_b_nesterov = self.beta1 * m_b_hat + (1 - self.beta1) * layer.grad_b
        
        # Update weights
        layer.W = layer.W - self.learning_rate * (m_W_nesterov / (np.sqrt(v_W_hat) + self.epsilon))
        layer.W = layer.W - self.learning_rate * self.weight_decay * layer.W
        layer.b = layer.b - self.learning_rate * (m_b_nesterov / (np.sqrt(v_b_hat) + self.epsilon))
    def step(self):
        """Increment the global time-step counter once per training batch."""
        self.t += 1

    def get_name(self):
        return "nadam"

def get_optimizer(optimizer_name,learning_rate=0.01,weight_decay=0.0):
    optimizer_dict = {
        'sgd': SGD,
        'momentum': Momentum,
        'nag': NAG,
        'rmsprop': RMSprop,
        'adam': Adam,
        'nadam': Nadam
    }
    
    if optimizer_name not in optimizer_dict:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")  
    return optimizer_dict[optimizer_name](learning_rate=learning_rate, weight_decay=weight_decay)