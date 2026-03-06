"""
Neural Network Model class
Handles forward and backward propagation loops
"""

import numpy as np
import os
import json
from .neural_layer import DenseLayer
from .activations import get_activation
from .objective_functions import get_loss
from .optimizers import get_optimizer

class NeuralNetwork:

    def __init__(self,cli_args):
        hidden_sizes = list(cli_args.hidden_size)

        # Hyper-parameters
        self.input_size = int(getattr(cli_args,"input_size",784))
        self.output_size = int(getattr(cli_args,"output_size",10))
        self.hidden_sizes = hidden_sizes
        self.activation_name = getattr(cli_args,"activation","relu")
        self.loss_name = getattr(cli_args,"loss","cross_entropy")
        self.weight_init     = getattr(cli_args, 'weight_init', 'xavier')
        self.optimizer_name  = getattr(cli_args, 'optimizer', 'adam')
        self.learning_rate   = getattr(cli_args, 'learning_rate',0.001)
        self.weight_decay    = getattr(cli_args, 'weight_decay',0.0)
        self.gradient_clip   = getattr(cli_args, 'gradient_clip',5.0)

        
        # Build layer list
        layer_sizes = [self.input_size]+hidden_sizes+[self.output_size]
        self.layers = []
        self.activations=[]

        for i in range(len(layer_sizes)-1):
            self.layers.append(DenseLayer(layer_sizes[i],layer_sizes[i+1],self.weight_init))
            if i<len(layer_sizes)-2:
                self.activations.append(get_activation(self.activation_name))
        
        # loss and optimizer
        self.loss_fn = get_loss(self.loss_name)
        self.optimizer = get_optimizer(self.optimizer_name,self.learning_rate,self.weight_decay)

        # last_output: raw logits from most recent forward(); used by
        # train.py to read predictions without a second forward pass.
        self.last_output = None

        # Gradient arrays (set by backward(); index 0 = last/output layer)
        self.grad_W = None
        self.grad_b = None

        # Per-layer gradient norm history (reset each epoch by train.py)
        self.gradient_norms   = {i: [] for i in range(len(self.layers))}
        # Per-neuron activation stats for symmetry analysis
        self.activation_stats = {i: [] for i in range(len(self.layers))}

    #forward / backward

    def forward(self,input):
        A=input
        for i,layer in enumerate(self.layers):
            Z = layer.forward(A)
            if i< len(self.activations):  # hidden layer
                A = self.activations[i].forward(Z)
            else:
                A =Z
        self.last_output = A
        return A
    
    def backward(self,y_true,y_pred):
        N = y_pred.shape[0]

        # Output layer gradient: combined loss + output activation
        if self.loss_name=="cross_entropy":
            Z_shifted = y_pred- y_pred.max(axis=1,keepdims=True)
            exp_z = np.exp(Z_shifted)
            probs= exp_z/exp_z.sum(axis=1,keepdims=True)
            dL_dZ_out = (probs-y_true)/N
    
        else:
            # MSE + Identity: dL/dZ = (2/N)(Z − y_true)
            dL_dZ_out = (2.0 / N) * (y_pred - y_true)
        grad_W_list = []
        grad_b_list = []
        # Backprop through output layer
        out_idx = len(self.layers)-1
        dL_dA = self.layers[out_idx].backward(dL_dZ_out)

        grad_W_list.append(self.layers[out_idx].grad_W)
        grad_b_list.append(self.layers[out_idx].grad_b)

        self.gradient_norms[out_idx].append(float(np.linalg.norm(self.layers[out_idx].grad_W)))

        # backprop through hidden layers (reverse)
        for i in reversed(range(len(self.layers)-1)):
            dL_dZ = self.activations[i].backward(dL_dA)
            dL_dA = self.layers[i].backward(dL_dZ)
            grad_W_list.append(self.layers[i].grad_W)
            grad_b_list.append(self.layers[i].grad_b)
            self.gradient_norms[i].append(float(np.linalg.norm(self.layers[i].grad_W)))

            # Per-neuron mean absolute gradient (first 5 neurons; for W&B symmetry plot)
            self.activation_stats[i].append(np.abs(self.layers[i].grad_W).mean(axis=0)[:5].tolist())

        # ── Pack as numpy object arrays (index 0 = last/output layer) ─────────
        # Using explicit object arrays avoids numpy broadcasting across
        # differently-shaped gradient matrices.
        grad_W_list.reverse()
        grad_b_list.reverse()
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b


    # -------------------------------------------------------------------------
    # Weight update
    # -------------------------------------------------------------------------
    def update_weights(self):
        self.optimizer.step()
        for layer in self.layers:
            self.optimizer.update(layer)

    def train_step(self, X, Y):
        if self.optimizer_name == 'nag':
            return self.train_step_nag(X, Y)
        # Standard path: forward → loss → backward → clip → update
        logits = self.forward(X)
        loss   = self.compute_loss(logits, Y)
        self.backward(Y, logits)

        if self.gradient_clip > 0:
            self.clip_gradients()
        self.update_weights()
        return loss
    

    def train_step_nag(self, X, Y):
        """
        Nesterov Accelerated Gradient training step.
        """
        beta  = getattr(self.optimizer, 'beta', 0.9)

        saved = [(l.W.copy(), l.b.copy()) for l in self.layers]

        # move to lookahead position
        for layer in self.layers:
            layer_id = id(layer)
            vW  = self.optimizer.velocities_W.get(layer_id, np.zeros_like(layer.W))
            vb  = self.optimizer.velocities_b.get(layer_id, np.zeros_like(layer.b))
            layer.W = layer.W - beta * vW
            layer.b = layer.b - beta * vb

        # forward + backward at lookahead
        logits = self.forward(X)
        loss   = self.compute_loss(logits, Y)
        self.backward(Y, logits)

        # restore original weights
        for layer, (W0, b0) in zip(self.layers, saved):
            layer.W = W0
            layer.b = b0

        if self.gradient_clip > 0:
            self.clip_gradients()

        # Step 5 — update with the lookahead gradients
        self.update_weights()
        return loss
    
    def train(self,x_train,y_train,epochs=1,batch_size=32):
        # convert integer labels to one hot if not done
        if y_train.ndim==1:
            one_hot = np.zeros((len(x_train),self.output_size))
            one_hot[np.arange(len(x_train)),y_train.astype(int)]=1
            y_train=one_hot
        
        history = {"loss":[]}

        for epoch in range(epochs):
            for k in self.gradient_norms:
                self.gradient_norms[k]=[]
            for k in self.activation_stats:
                self.activation_stats[k]=[]
            
            indexes = np.random.permutation(len(x_train))
            epoch_losses=[]
            for start in range(0,len(indexes),batch_size):
                batch = indexes[start:start+batch_size]
                epoch_losses.append(self.train_step(x_train[batch],y_train[batch]))
            history["loss"].append(float(np.mean(epoch_losses)))
        return history

    # inference 

    def predict(self,input):
        logits = self.forward(input)
        z_shifted = logits - logits.max(axis=1,keepdims=True)
        exp_z = np.exp(z_shifted)
        probs = exp_z/exp_z.sum(axis=1,keepdims=True)
        preds = np.argmax(probs ,axis=1)
        return preds,probs
    
    def compute_loss(self,logits,y_true):
        if self.loss_name=="cross_entropy":
            z_shifted = logits-logits.max(axis=1,keepdims=True)
            exp_z = np.exp(z_shifted)
            probs = exp_z/exp_z.sum(axis=1,keepdims=True)
            return self.loss_fn.forward(probs,y_true)
        else:
            return self.loss_fn.forward(logits,y_true)
    
    def evaluate(self,input,label):
        logits = self.forward(input)
        # Apply softmax for consistency
        z_shifted = logits - logits.max(axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        probs = exp_z / exp_z.sum(axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)
        y_int = label if label.ndim==1 else np.argmax(label,axis=1)
        return float(np.mean(preds==y_int))


    def clip_gradients(self):
        total_sq = sum(
            np.sum(layer.grad_W**2) + np.sum(layer.grad_b**2)
            for layer in self.layers
            if layer.grad_W is not None
        )
        global_norm = np.sqrt(total_sq+1e-10)

        if global_norm>self.gradient_clip:
            coef = self.gradient_clip/global_norm
            for layer in self.layers:
                if layer.grad_W is not None and layer.grad_b is not None:
                    layer.grad_W*=coef
                    layer.grad_b*=coef
    


    def get_weights(self):
        d = {}
        for i,layer in enumerate(self.layers):
            d[f"W{i}"]=layer.W.copy()
            d[f"b{i}"]=layer.b.copy()
        return d
    
    def set_weights(self,weight_dict):
        for i,layer in enumerate(self.layers):
            if f"W{i}" in weight_dict:
                layer.W = weight_dict[f"W{i}"].copy()
            if f"b{i}" in weight_dict:
                layer.b = weight_dict[f"b{i}"].copy()
    
    def save_weights(self,filepath):
        """Save weights as dictionary in .npy file format."""
        weights_dict = self.get_weights()
        np.save(filepath, weights_dict)
    
    def load_weights(self, filepath):
        """Load weights from a .npy file containing a dictionary."""
        weights_dict = np.load(filepath, allow_pickle=True).item()
        self.set_weights(weights_dict)

    def get_config(self):
        """Return architecture and hyper-parameters as a JSON-serialisable dict."""
        return {
            'input_size':    self.input_size,
            'hidden_sizes':  self.hidden_sizes,
            'output_size':   self.output_size,
            'activation':    self.activation_name,
            'loss':          self.loss_name,
            'weight_init':   self.weight_init,
            'optimizer':     self.optimizer_name,
            'learning_rate': float(self.learning_rate),
            'weight_decay':  float(self.weight_decay),
            'gradient_clip': float(self.gradient_clip),
        }


    def save_config(self,filepath):
        with open(filepath,'w') as file:
            json.dump(self.get_config(),file,indent=4)
    
    def save(self,weights_path,config_path):
        os.makedirs(os.path.dirname(os.path.abspath(weights_path)),exist_ok=True)
        self.save_weights(weights_path)
        self.save_config(config_path)
    
    def get_layer_output(self,input,layer_idx):
        """Return the post-activation output of a specific hidden layer."""
        A = input
        for i in range(layer_idx+1):
            Z = self.layers[i].forward(A)
            if i<len(self.activations):
                A = self.activations[i].forward(Z)
            else:
                A =Z
        return A
    
    def get_dead_neurons(self,input,threshold=0.01):
        """
        Identify RELU neurons that never activate.
        Only reports hidden layers with RELU activations; skip others.
        
        """
        dead_info = {}
        A = input

        for idx, (layer, act) in enumerate(zip(self.layers[:-1], self.activations)):

            Z = layer.forward(A)
            A = act.forward(Z)

            if act.get_name().lower() == "relu":

                activation_rate = np.sum(A > 0, axis=0) / A.shape[0]
                dead_neurons = np.where(activation_rate < threshold)[0]

                dead_info[idx] = {
                    "dead_neurons": dead_neurons.tolist(),
                    "num_dead": len(dead_neurons),
                    "activation_rates": activation_rate.tolist()
                }

        return dead_info