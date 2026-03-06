"""
Main Training Script
Entry point for training neural networks with CLI arguments.

W&B logging covers ALL 10 report sections:
  Section 2.1  Data Exploration  -- 5 sample images per class + class distribution bar chart
  Section 2.2  Hyperparameter Sweep -- wandb.sweep() with 100+ runs, parallel coords
  Section 2.3  Optimizer Showdown  -- train/val loss & accuracy every epoch
  Section 2.4  Vanishing Gradient  -- per-layer gradient norm mean/max every epoch
  Section 2.5  Dead Neuron        -- dead-neuron count/% per hidden layer every epoch
  Section 2.6  Loss Comparison    -- explicit loss_comparison/ keys for MSE vs CE overlay
  Section 2.7  Global Performance -- train/val/test accuracy as run summary for scatter plot
  Section 2.8  Error Analysis     -- confusion matrix on test set
  Section 2.9  Weight Init/Symmetry -- per-neuron gradients for first 50 steps
  Section 2.10 Fashion-MNIST      -- use --dataset fashion_mnist flag 
"""
import json

import argparse
import numpy as np
import os
import wandb

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data,one_hot,DataLoader,get_sample_images,get_class_distribution
from utils.metrics import evaluate_model,print_metrics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
# Reproducibilty
def set_seed(seed=42):
    np.random.seed(seed)

# Section 2.1: Data Exploration & Class Distribution
def log_data_exploration(x_train,y_train,dataset_name):
    """Log a W&B table containing 5 sample images from each of the 10 classes."""

    FASHION_LABELS = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker",
                      "Bag","Ankle Boot"]
    MNIST_LABELS = [str(i) for i in range(10)]
    class_names =FASHION_LABELS if dataset_name=="fashion_mnist" else MNIST_LABELS
    # Sample images table
    sample_images,sample_labels = get_sample_images(x_train,y_train,samples_per_class=5,num_classes=10)

    img_table = wandb.Table(columns=["class_id","class_name","image"])
    for class_id,images in sample_images.items():
        for img in images:
            img = img.reshape(28,28)
            img_table.add_data(class_id,class_names[class_id],wandb.Image(
                img,caption=f"Class {class_id}: {class_names[class_id]}"
            ))
    wandb.log({"data_exploration/sample_images":img_table})
    
    # class distribution bar chart
    distribution,percentages = get_class_distribution(y_train,num_classes=10)
    dist_table = wandb.Table(columns=["class_id","class_name","count","percentage"])
    for cid in range(10):
        dist_table.add_data(cid,class_names[cid],int(distribution[cid]),round(float(percentages[cid]),2))
    
    wandb.log({"data_exploration/class_distribution":wandb.plot.bar(dist_table,"class_name","count",
                                                                    title="Class Distribution -- Training Set")})
    
    print("Logged Sample images and class distribution to W&B")


# Section 2.4 -- Vanishing Gradient Analysis

def log_gradient_norms(model,epoch):
    """Log mean and max gradient norm for every layer."""

    log_dict = {"epoch":epoch}
    for layer_idx,norms in model.gradient_norms.items():
        if norms:
            log_dict[f"gradient/layer_{layer_idx}_norm_mean"]=float(np.mean(norms))
            log_dict[f"gradient/layer_{layer_idx}_norm_max"] = float(np.max(norms))

    wandb.log(log_dict)

# Section 2.9 -- Weight Initialization & Symmetry

def log_per_neuron_gradients(model,global_step,num_neurons=5):
    if global_step>=50:
        return
    first_layer =  model.layers[0]
    if first_layer.grad_W is None:
        return
    log_dict = {"global_step":global_step}
    n = min(num_neurons,first_layer.grad_W.shape[1])
    for i in range(n):
        log_dict[f'symmetry/layer0_neuron{i}_grad'] = float(
            np.mean(np.abs(first_layer.grad_W[:, i]))
        )
    wandb.log(log_dict)

# =============================================================================
# Section 2.5 -- Dead Neuron Investigation  (6 Marks)
# =============================================================================

def log_dead_neurons(model,x_probe):
    """
    Log dead-neuron count and % per hidden layer (Section 2.5).

    A neuron is dead if it outputs 0 for more than 99% of probe samples.
    Only ReLU layers produce dead neurons; other activations are skipped.
    """
    dead_info = model.get_dead_neurons(x_probe, threshold=0.01)
    if not dead_info:
        return
    log_dict = {}
    for layer_idx, info in dead_info.items():
        total = len(info['activation_rates'])
        log_dict[f'dead_neurons/layer_{layer_idx}_count'] = info['num_dead']
        if total > 0:
            log_dict[f'dead_neurons/layer_{layer_idx}_pct'] = (
                info['num_dead'] / total * 100.0
            )
    wandb.log(log_dict)

# =============================================================================
# Section 2.6 -- Loss Function Comparison  (4 Marks)
# =============================================================================

def log_loss_comparison(epoch, train_loss, val_loss, loss_fn_name):
    """
    Log training and validation loss under a dedicated namespace that includes
    the loss function name (Section 2.6).

    Grouping runs by 'loss_function' in the W&B report overlays the MSE and
    CE learning curves on the same plot.

    """
    wandb.log({
        "loss_comparison/epoch":      epoch,
        "loss_comparison/train_loss": float(train_loss),
        "loss_comparison/val_loss":   float(val_loss),
        "loss_function":              loss_fn_name,
    })

# =============================================================================
# Section 2.7 -- Global Performance Analysis  (4 Marks)
# =============================================================================

def log_global_performance(train_acc, val_acc, test_acc):
    """
    Log train / val / test accuracy as run summary values (Section 2.7).

    W&B creates a scatter plot across all runs when these summary keys exist,
    enabling the Training vs Test Accuracy overlay that identifies overfitting.
    """
    wandb.log({
        "performance/final_train_accuracy": float(train_acc),
        "performance/final_val_accuracy":   float(val_acc),
        "performance/test_accuracy":        float(test_acc),
    })
    # Also write to run summary so they appear in the sweep parallel-coords plot
    wandb.run.summary["train_accuracy"] = float(train_acc)
    wandb.run.summary["val_accuracy"]   = float(val_acc)
    wandb.run.summary["test_accuracy"]  = float(test_acc)
    print("  OK  Logged global performance (train/val/test accuracy) to W&B")


# =============================================================================
# Section 2.8 -- Error Analysis  (5 Marks)
# =============================================================================

def log_confusion_matrix(test_preds, test_labels, class_names):
    """
    Log the confusion matrix to W&B (Section 2.8).
    """
    wandb.log({
        "error_analysis/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=test_labels.tolist(),
            preds=test_preds.tolist(),
            class_names=class_names
        )
    })
    print("  OK  Logged confusion matrix to W&B")




# Section 2.2 -- Hyperparameter Sweep configuration

sweep_config = {
    "method":"bayes",
    "metric":{
        "name":"val/accuracy",
        "goal":"maximize"
    },
    "parameters":{
        "learning_rate":{
            "distribution":"log_uniform_values",
            "min":1e-4,
            "max":1e-2
        },
        "batch_size":{
            "values": [16,32,64,128]
        },
        "optimizer":{
            "values":["sgd","momentum","nag","rmsprop","adam","nadam"]
        },
        "activation":{
            "values":["sigmoid","tanh","relu"]
        },
        "weight_init":{
            "values":["random","xavier"]
        },
        "hidden_size_cfg":{
            "values": [
                "128",
                "128-64",
                "128-32",
                "128-64-32",
                "128-128-64",
                "64-64-64",
                "64-32"
            ]
        },
        "weight_decay":{
            "values":[0.0,1e-4,1e-3]
        },
        "loss":{
            "values":["cross_entropy","mse"]
        }
    }
}

def sweep_train():
    """
    wandb.agent() calls this function for each trial. 
    It reads the sweep configuration from wandb.config, builds an args namespace, then call the shared train loop.
    """
    with wandb.init() as run:
        cfg = wandb.config

        # Decode hidden_size_config string into list[int]
        hidden_size = [int(x) for x in cfg.hidden_size_cfg.split("-")]

        class SweepArgs:
            pass

        args = SweepArgs()
        args.dataset = "mnist"
        args.epochs = 20
        args.batch_size =cfg.batch_size
        args.loss = cfg.loss
        args.optimizer = cfg.optimizer
        args.learning_rate = cfg.learning_rate
        args.weight_decay = cfg.weight_decay
        args.hidden_size = hidden_size
        args.num_layers = len(hidden_size)
        args.activation = cfg.activation
        args.weight_init = cfg.weight_init
        args.gradient_clip = 1.0
        args.wandb_project = run.project
        args.model_save_path = "models"
        args.input_size=784
        args.output_size=10
        train_inner(args,run_already_init=True)

def run_sweep(project_name,sweep_count=100):
    sweep_id = wandb.sweep(sweep_config,project=project_name)
    print(f"\n Sweep created: {sweep_id}")
    print(f"Starting {sweep_count} agents ...\n")
    wandb.agent(sweep_id,function=sweep_train,count=sweep_count)


# Training Loop 
def train_inner(args,run_already_init=False):
    """Full Training loop used by both normal training and sweep."""

    set_seed(42)

    # W&B init (skipped when sweep already did it)
    if not run_already_init:
        wandb_config = {
            "dataset": args.dataset,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "loss": args.loss,
            "optimizer":  args.optimizer,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip":  args.gradient_clip,
            "num_hidden_layers": args.num_layers,
            "hidden_sizes": args.hidden_size,
            "activation": args.activation,
            "weight_init": args.weight_init,
        }

        wandb.init(
            project=args.wandb_project,
            config = wandb_config,
            tags=[args.dataset, args.optimizer,args.activation,args.weight_init],
            notes=f"{args.num_layers} hidden layers, init={args.weight_init}",
        )
    print("Training Neural Network")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Optimizer: {args.optimizer} | LR: {args.learning_rate}")
    print(f"Activation: {args.activation} | Loss: {args.loss}")
    print(f"Init: {args.weight_init}")
    print("="*60 + "\n")

    # Load data
    x_train,x_val,x_test,y_train,y_val,y_test = load_data(args.dataset)

    y_train_oh = one_hot(y_train,num_classes=10)
    y_val_oh = one_hot(y_val,num_classes=10)
    y_test_oh = one_hot(y_test,num_classes=10)

    train_loader = DataLoader(x_train,y_train_oh,batch_size=args.batch_size,shuffle=True)
    val_loader =  DataLoader(x_val,y_val_oh,batch_size=args.batch_size,shuffle=False)
    test_loader = DataLoader(x_test,y_test_oh,batch_size=args.batch_size,shuffle=False)


    val_probe = x_val[:256]

    FASHION_LABELS = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal",      "Shirt",   "Sneaker",  "Bag",   "Ankle boot"
    ]
    class_names = (FASHION_LABELS if args.dataset == 'fashion_mnist'
                   else [str(i) for i in range(10)])

    # Data Exploration (once, before training)
    print("Logging data exploration to W&B")
    log_data_exploration(x_train,y_train,args.dataset)

    # Build model
    model = NeuralNetwork(args)
    print(f"\n Model: \n {model}\n")

    best_val_f1      = 0.0
    best_val_acc     = 0.0
    best_epoch       = 0
    patience         = 20
    patience_counter = 0
    global_step      = 0

    for epoch in range(args.epochs):
        for key in model.gradient_norms:
            model.gradient_norms[key]=[]
        for key in model.activation_stats:
            model.activation_stats[key]=[]
        

        # Train
        train_losses=[]
        train_preds =[]
        train_labels_list=[]

        for x_batch,y_batch in train_loader:
            loss = model.train_step(x_batch,y_batch)
            train_losses.append(loss)

            # per neuron gradients
            log_per_neuron_gradients(model,global_step,num_neurons=5)
            global_step+=1

            train_preds.extend(np.argmax(model.last_output,axis=1))
            train_labels_list.extend(np.argmax(y_batch, axis=1))
        
        # Validate
        val_preds =[]
        val_labels_list=[]
        val_losses=[]

        for x_batch,y_batch in val_loader:
            logits = model.forward(x_batch)
            val_losses.append(model.compute_loss(logits,y_batch))
            val_preds.extend(np.argmax(logits,axis=1))
            val_labels_list.extend(np.argmax(y_batch,axis=1))

        train_preds_arr = np.array(train_preds)
        train_labels_arr = np.array(train_labels_list)
        val_preds_arr    = np.array(val_preds)
        val_labels_arr   = np.array(val_labels_list)

        train_metrics = evaluate_model(train_preds_arr, train_labels_arr)
        val_metrics   = evaluate_model(val_preds_arr,   val_labels_arr)
        train_loss    = float(np.mean(train_losses))
        val_loss      = float(np.mean(val_losses))

         # Early stopping
        if val_metrics['f1'] > best_val_f1:
            best_val_f1      = val_metrics['f1']
            best_val_acc     = val_metrics['accuracy']
            best_epoch       = epoch
            patience_counter = 0
            os.makedirs(MODELS_DIR, exist_ok=True)
            model.save(
                os.path.join(MODELS_DIR, f'best_model_{args.dataset}.npy'),
                os.path.join(MODELS_DIR, f'best_config_{args.dataset}.json')
            )
        else:
            patience_counter += 1

        # W&B logging ──────────────────────────────────────────────────────────
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_metrics["accuracy"],
            'train/precision': train_metrics['precision'],
            'train/recall':    train_metrics['recall'],
            'train/f1':        train_metrics['f1'],
            'val/loss':        val_loss,
            'val/accuracy':    val_metrics['accuracy'],
            'val/precision':   val_metrics['precision'],
            'val/recall':      val_metrics['recall'],
            'val/f1':          val_metrics['f1'],
        })

        # Section 2.6 -- loss comparison (MSE vs CE overlay)
        log_loss_comparison(epoch, train_loss, val_loss, args.loss)

        # Section 2.4 -- gradient norms
        log_gradient_norms(model, epoch)

        # Section 2.5 -- dead neurons
        log_dead_neurons(model, val_probe)

        if (epoch + 1) % 5 == 0:
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            print(f"  Train Loss: {train_loss:.6f}  |  Val Loss: {val_loss:.6f}")
            print(f"  Train Acc:  {train_metrics['accuracy']:.4f}  "
                  f"|  Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"  Train F1:   {train_metrics['f1']:.4f}  "
                  f"|  Val F1:  {val_metrics['f1']:.4f}")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1} "
                  f"(no improvement for {patience} epochs)")
            break



    # ─────────────────────────────────────────────────────────────────
    # LOAD BEST MODEL FROM CHECKPOINT
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LOADING BEST MODEL FROM CHECKPOINT")
    print("=" * 60 + "\n")
    
    checkpoint_path = os.path.join(MODELS_DIR, f'best_model_{args.dataset}.npy')
    checkpoint_config = os.path.join(MODELS_DIR, f'best_config_{args.dataset}.json')
    

    if os.path.exists(checkpoint_path) and os.path.exists(checkpoint_config):

        # Load saved architecture
        with open(checkpoint_config, "r") as f:
            saved_config = json.load(f)

        args.hidden_size = saved_config["hidden_sizes"]
        args.num_layers = len(args.hidden_size)
        # Rebuild model using saved architecture
        model = NeuralNetwork(args)

        # Load weights
        model.load_weights(checkpoint_path)

        print("✓ Loaded best model from checkpoint")
        print(f"  Path: {checkpoint_path}")
        print(f"  Best validation F1: {best_val_f1:.4f}")
        print(f"  Best validation Accuracy: {best_val_acc:.4f}")

    else:
        print(f"⚠ WARNING: Checkpoint not found at {checkpoint_path}")
        print(f"  Using current model (may not be optimal)")

    # Test phase - USING BEST MODEL
    print("\n" + "=" * 60)
    print("TESTING BEST MODEL ON HELD-OUT TEST SET")
    print("=" * 60 + "\n")

    test_preds_list  = []
    test_labels_list = []

    for X_batch, Y_batch in test_loader:
        logits = model.forward(X_batch)
        test_preds_list.extend(np.argmax(logits, axis=1))
        test_labels_list.extend(np.argmax(Y_batch, axis=1))

    test_preds_arr  = np.array(test_preds_list)
    test_labels_arr = np.array(test_labels_list)

    test_metrics = evaluate_model(test_preds_arr, test_labels_arr)
    print_metrics(test_metrics, "Test")
    print(f"\nBest val F1: {best_val_f1:.4f}  (Epoch {best_epoch + 1})")

    wandb.log({
        'test/accuracy':  test_metrics['accuracy'],
        'test/precision': test_metrics['precision'],
        'test/recall':    test_metrics['recall'],
        'test/f1':        test_metrics['f1'],
    })

    final_train_acc = float(train_metrics['accuracy'])
    log_global_performance(final_train_acc, best_val_acc, test_metrics['accuracy'])
    log_confusion_matrix(test_preds_arr, test_labels_arr, class_names)

    # ─────────────────────────────────────────────────────────────────
    # SAVE BEST MODEL TO FINAL LOCATION
    # ─────────────────────────────────────────────────────────────────
    save_dir = MODELS_DIR
    os.makedirs(save_dir, exist_ok=True)

    model.save(
        os.path.join(save_dir, f'best_model_{args.dataset}.npy'),
        os.path.join(save_dir, f'best_config_{args.dataset}.json')
    )
    
    print(f"\n" + "=" * 60)
    print("✓ MODEL SAVED SUCCESSFULLY")
    print("=" * 60)
    print(f"Best model path: {os.path.join(save_dir, f'best_model_{args.dataset}.npy')}")
    print(f"Config path:     {os.path.join(save_dir, f'best_config_{args.dataset}.json')}")
    print(f"\nTest Performance (Best Model):")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1']:.4f}")
    print("=" * 60)

    if not run_already_init:
        wandb.finish()

    return model, test_metrics


def train(args):
    return train_inner(args,run_already_init=False)

# =============================================================================
# CLI argument parser
# =============================================================================


def parse_arguments():
    """Parse command-line arguments for training."""

    parser = argparse.ArgumentParser(
        description="Train a Neural network on MNIST / Fashion-MNIST"
    )

    # Specifying arguments
    parser.add_argument('-d','--dataset',type=str,default='mnist',choices=["mnist","fashion_mnist"],
                        help="Dataset to train on")
    parser.add_argument('-e',   '--epochs',        type=int,   default=50,
                        help='Number of training epochs')

    parser.add_argument('-b',   '--batch_size',    type=int,   default=32,
                        help='Mini-batch size')

    parser.add_argument('-l',   '--loss',          type=str,   default='cross_entropy',
                        choices=['mse', 'cross_entropy'],
                        help='Loss function')

    parser.add_argument('-o',   '--optimizer',     type=str,   default='adam',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],
                        help='Optimisation algorithm')

    parser.add_argument('-lr',  '--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')

    parser.add_argument('-wd',  '--weight_decay',  type=float, default=0.0,
                        help='L2 regularisation (weight decay) strength')

    parser.add_argument('-nhl', '--num_layers',    type=int,   default=2,
                        help='Number of hidden layers (must match len of --hidden_size)')

    parser.add_argument('-sz',  '--hidden_size',   type=int,   nargs='+',
                        default=[128, 64],
                        help='Neuron count for each hidden layer e.g. -sz 128 128 64')

    parser.add_argument('-a',   '--activation',    type=str,   default='relu',
                        choices=['sigmoid', 'tanh', 'relu', 'leaky_relu'],
                        help='Hidden-layer activation function')

    parser.add_argument('-wi',  '--weight_init',   type=str,   default='xavier',
                        choices=['random', 'xavier', 'he', 'zeros'],
                        help='Weight initialisation strategy')

    # Extra arguments
    parser.add_argument('-gc',  '--gradient_clip', type=float, default=1.0,
                        help='Global-norm gradient clipping threshold (0 = disabled)')

    parser.add_argument('--wandb_project',   type=str, default='da6401_assignment_1',
                        help='W&B project name')

    parser.add_argument('--model_save_path', type=str, default='models',
                        help='Relative directory to save trained model')

    # Section 2.2 -- sweep flags
    parser.add_argument('--sweep',       action='store_true',
                        help='Run hyperparameter sweep (Section 2.2)')
    parser.add_argument('--sweep_count', type=int, default=100,
                        help='Number of sweep trials (default 100, min 100 required)')

    args = parser.parse_args()

    # Ensure compatibility between number of layers and hidden sizes
    if len(args.hidden_size) != args.num_layers:
        raise ValueError(
            f"Mismatch between --num_layers and --hidden_size.\n"
            f"num_layers = {args.num_layers}, but hidden_size list = {args.hidden_size}\n"
            f"Please ensure len(hidden_size) == num_layers.\n"
            f"Example: -nhl 3 -sz 128 128 64"
        )

    args.input_size  = 784
    args.output_size = 10

    return args

# =============================================================================
# Entry point
# =============================================================================

def main():
    args = parse_arguments()

    if args.sweep:
        # Section 2.2 -- Hyperparameter sweep
        print(f"\nStarting W&B hyperparameter sweep ({args.sweep_count} trials) ...")
        run_sweep(args.wandb_project, sweep_count=args.sweep_count)
    else:
        # Normal single training run
        train(args)


if __name__ == '__main__':
    main()