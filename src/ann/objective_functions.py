"""
Loss Functions for Neural Network
Implements: Mean Squared Error (MSE) and Cross-Entropy Loss
"""

import numpy as np

class Loss:
    """Base class for loss functions"""
    def forward(self, y_pred, y_true):
        raise NotImplementedError
    
    def backward(self, y_pred, y_true):
        raise NotImplementedError
    
    def get_name(self):
        raise NotImplementedError

class MSE(Loss):
    def forward(self,y_pred,y_true):
        batch_size  = y_pred.shape[0]
        squared_error = (y_pred-y_true)**2
        loss = np.sum(squared_error)/batch_size
        return loss
    
    def backward(self, y_pred, y_true):
        """
        ✅ FIXED: Following Prasad's MSE backward
        Returns: 2 * (y_pred - y_true) / batch_size
        with the softmax jacobian transformation
        """
        n = len(y_pred)
        grad = 2.0 * (y_pred - y_true) / n
        tmp = np.sum(grad * y_pred, axis=1, keepdims=True)
        return y_pred * (grad - tmp)
    
    def get_name(self):
        return "mse"

class CrossEntropy(Loss):
    def forward(self,y_pred,y_true):
        batch_size = y_pred.shape[0]
        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-10, 1.0)     
        # Cross-entropy: -Σ(Y_true * log(Y_pred))
        loss = -np.sum(y_true * np.log(y_pred_clipped)) / batch_size
        
        return loss
    
    def backward(self, y_pred, y_true):
        """
        Simple formula: (y_pred - y_true) / batch_size
        
        This is the gradient of cross-entropy + softmax combined
        """
        batch_size = len(y_pred)
        return (y_pred - y_true) / batch_size
    
    def get_name(self):
        return "cross_entropy"

def get_loss(loss_name):
    loss_dict = {
        'mse': MSE,
        'cross_entropy': CrossEntropy
    }
    
    if loss_name not in loss_dict:
        raise ValueError(f"Unknown loss function: {loss_name}")
    
    return loss_dict[loss_name]()