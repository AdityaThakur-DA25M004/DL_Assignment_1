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
    def backward(self, Y_pred, Y_true):
        batch_size = Y_pred.shape[0]
        dL_dY_pred = (2.0 / batch_size) * (Y_pred - Y_true)
        
        return dL_dY_pred
    
    def get_name(self):
        return "mse"

class CrossEntropy(Loss):
    def forward(self,y_pred,y_true):
        batch_size = y_pred.shape[0]
        # Clip predictions to avoid log(0)
        Y_pred_clipped = np.clip(y_pred, 1e-10, 1.0)     
        # Cross-entropy: -Σ(Y_true * log(Y_pred))
        loss = -np.sum(y_true * np.log(Y_pred_clipped)) / batch_size
        
        return loss
    def backward(self,y_pred,y_true):
        batch_size = y_pred.shape[0]
        Y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
        dL_dA = -y_true / (Y_pred_clipped * batch_size)
        return dL_dA
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