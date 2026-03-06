# """
# Activation Functions for Neural Network
# Implements: Sigmoid, Tanh, ReLU, Softmax
# """

# import numpy as np

# class Activation:
#     """Base class for activation functions"""
#     def forward(self,Z):
#         raise NotImplementedError
#     def backward(self,dL_dA):
#         raise NotImplementedError
#     def get_name(self):
#         raise NotImplementedError

# class Sigmoid(Activation):
#     """
#     Sigmoid activation: sigma(z) = 1 / (1 + e^-z)
#     Range: (0, 1)
#     """
#     def forward(self,Z):
#         Z_clipped = np.clip(Z,-500,500)
#         # ✅ FIX: Copy the sigmoid output to prevent state aliasing
#         self.sigma = (1 / (1 + np.exp(-Z_clipped))).copy()
#         return self.sigma
    
#     def backward(self,dL_dsigma):
#         dL_dZ = dL_dsigma * self.sigma * (1 - self.sigma)
#         return dL_dZ 
    
#     def get_name(self):
#         return "sigmoid"

# class Tanh(Activation):
#     """
#     Tanh activation: tanh(z) = (e^z - e^-z) / (e^z + e^-z)
#     Range: (-1, 1)
#     """
#     def forward(self,Z):
#         # ✅ FIX: Copy the tanh output to prevent state aliasing
#         self.tanh = np.tanh(Z).copy()
#         return self.tanh
    
#     def backward(self, dL_dA):
#         dL_dZ = dL_dA * (1 - self.tanh**2)
#         return dL_dZ
    
#     def get_name(self):
#         return "tanh"

# class ReLu(Activation):
#     """
#     ReLU activation: max(0, z)
#     Range: [0, ∞)
#     """
#     def forward(self, Z):
#         self.Z = Z.copy()  # ✅ FIX: Copy input for backward pass
#         # ✅ FIX: Copy the output to prevent state aliasing
#         self.A = np.maximum(0, Z).copy()
#         return self.A
    
#     def backward(self, dL_dA):
#         dL_dZ = dL_dA * (self.Z > 0).astype(float)
#         return dL_dZ
    
#     def get_name(self):
#         return "relu"

# class LeakyReLu(Activation):
#     """
#     Leaky ReLU with learnable slope alpha
#     """
#     def __init__(self, alpha=0.01):
#         self.alpha = alpha
    
#     def forward(self, Z):
#         self.Z = Z.copy()  # ✅ FIX: Copy input for backward pass
#         # ✅ FIX: Copy the output to prevent state aliasing
#         self.A = np.where(Z > 0, Z, self.alpha * Z).copy()
#         return self.A
    
#     def backward(self, dL_dA):
#         # Gradient is 1 where Z > 0, alpha elsewhere
#         dL_dZ = dL_dA * np.where(self.Z > 0, 1, self.alpha)
#         return dL_dZ

#     def get_name(self):
#         return "leaky_relu"

# class Softmax(Activation):
#     """
#     Softmax activation: p_i = exp(z_i) / sum(exp(z_j))
#     Range: (0, 1) per element, sum=1 per sample
    
#     Note: Softmax is NOT used as hidden layer activation in this network.
#     Output layer uses combined softmax+cross-entropy in NeuralNetwork.backward()
#     """

#     def forward(self,Z):
#         # Subtract max for numerical stability (prevents overflow)
#         Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
#         exp_Z = np.exp(Z_shifted)
#         # ✅ FIX: Copy the softmax output to prevent state aliasing
#         self.A = (exp_Z / np.sum(exp_Z, axis=1, keepdims=True)).copy()
#         return self.A
    
#     def backward(self, dL_dA):
#         batch_size = self.A.shape[0]
#         dL_dZ = np.zeros_like(self.A)
#         for i in range(batch_size):
#             jacobian = np.diagflat(self.A[i]) - np.outer(self.A[i], self.A[i])
#             dL_dZ[i] = np.dot(jacobian, dL_dA[i])
#         return dL_dZ
    
#     def get_name(self):
#         return "softmax"


# def get_activation(activation_name):
#     activation_dict = {
#         'sigmoid': Sigmoid,
#         'tanh': Tanh,
#         'relu': ReLu,
#         'leaky_relu': LeakyReLu,
#         'softmax': Softmax
#     }
    
#     if activation_name not in activation_dict:
#         raise ValueError(f"Unknown activation: {activation_name}. Choose from: {list(activation_dict.keys())}")
    
#     return activation_dict[activation_name]()


"""
Activation Functions for Neural Network
Implements: Sigmoid, Tanh, ReLU, Softmax
"""

import numpy as np

class Activation:
    """Base class for activation functions"""
    def forward(self,Z):
        raise NotImplementedError
    def backward(self,dL_dA):
        raise NotImplementedError
    def get_name(self):
        raise NotImplementedError

class Sigmoid(Activation):
    """
    Sigmoid activation: sigma(z) = 1 / (1 + e^-z)
    Range: (0, 1)
    """
    def forward(self,Z):
        Z_clipped = np.clip(Z,-500,500)
        self.sigma = 1 / (1 + np.exp(-Z_clipped))
        return self.sigma
    def backward(self,dL_dsigma):
        dL_dZ = dL_dsigma* self.sigma*(1-self.sigma)
        return dL_dZ 
    def get_name(self):
        return "sigmoid"

class Tanh(Activation):
    """
    Tanh activation: tanh(z) = (e^z - e^-z) / (e^z + e^-z)
    Range: (-1, 1)
    """
    def forward(self,Z):
        self.tanh = np.tanh(Z)
        return self.tanh
    def backward(self, dL_dA):
        dL_dZ = dL_dA * (1-self.tanh**2)
        return dL_dZ
    def get_name(self):
        return "tanh"

class ReLu(Activation):
    """
    ReLU activation: max(0, z)
    Range: [0, ∞)
    """
    def forward(self, Z):
        self.Z =Z
        self.A = np.maximum(0,Z)
        return self.A
    def backward(self, dL_dA):
        dL_dZ = dL_dA*(self.Z>0).astype(float)
        return dL_dZ
    def get_name(self):
        return "relu"
class LeakyReLu(Activation):
    def __init__(self,alpha=0.01):
        self.alpha= alpha
    def forward(self,Z):
        self.Z =Z
        self.A = np.where(Z>0,Z,self.alpha*Z)
        return self.A
    def backward(self,dL_dA):
        # Gradient is 1 where Z > 0, alpha elsewhere
        dL_dZ = dL_dA * np.where(self.Z > 0, 1, self.alpha)
        return dL_dZ

    def get_name(self):
        return "leaky_relu"
class Softmax(Activation):

    def forward(self,Z):
        # Subtract max for numerical stability (prevents overflow)
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        self.A = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        return self.A
    def backward(self, dL_dA):
        batch_size = self.A.shape[0]
        dL_dZ = np.zeros_like(self.A)
        for i in range(batch_size):
            jacobian = np.diagflat(self.A[i])-np.outer(self.A[i],self.A[i])
            dL_dZ[i] = np.dot(jacobian,dL_dA[i])
        return dL_dZ
    def get_name(self):
        return "softmax"


def get_activation(activation_name):
    activation_dict = {
        'sigmoid': Sigmoid,
        'tanh': Tanh,
        'relu': ReLu,
        'leaky_relu': LeakyReLu,
        'softmax': Softmax
    }
    
    if activation_name not in activation_dict:
        raise ValueError(f"Unknown activation: {activation_name}. Choose from: {list(activation_dict.keys())}")
    
    return activation_dict[activation_name]()