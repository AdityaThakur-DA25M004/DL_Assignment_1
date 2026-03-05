"""
Implements forward and backward passes for a linear layer
"""
import numpy as np

class DenseLayer:
    def __init__(self,input_size,output_size,weight_init="xavier"):
        self.input_size=input_size
        self.output_size=output_size

        # initialize weights based on method
        if weight_init=="xavier":
            limit = np.sqrt(6.0/(input_size+output_size))
            self.W = np.random.uniform(low=-limit,high=limit,size=(input_size,output_size))
        elif weight_init=="random":
            self.W = np.random.uniform(-0.5,0.5,(input_size,output_size)) 
        elif weight_init=="he":
            std = np.sqrt(2.0/input_size)
            self.W = np.random.normal(0,std,(input_size,output_size))
        elif weight_init=="zeros":
            self.W = np.zeros((input_size, output_size))
        else:
            raise ValueError(f"Unknown initialization:{weight_init}. Valid initializations are: 'xavier','he','random'")
        self.b = np.zeros((1,output_size))
        self.grad_W = None
        self.grad_b = None
        self.X = None

    def forward(self,input):
        """
        Forward pass:
        Z = XW + b
        """
        self.X=input
        self.Z = input @ self.W +self.b
        return self.Z # pass to next layer
    def backward(self,dL_dZ):
        """
        Backward pass.
        Args:
            dL_dZ : Gradient of loss w.r.t. layer output (Z)
        Returns:
            dL_dX : Gradient of loss w.r.t. layer input (X)
        """
        self.grad_W = self.X.T @ dL_dZ
        self.grad_b = np.sum(dL_dZ,axis=0,keepdims=True)
        dL_dX = dL_dZ @ self.W.T
        return dL_dX  # pass to previous layer
    def get_weights(self):
        return self.W,self.b
    def get_gradients(self):
        return self.grad_W,self.grad_b
    def set_weights(self,W,b):
        self.W = W.copy()
        self.b = b.copy()     