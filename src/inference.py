"""
Load a Trained model and evaluate on test dataset
Outputs: Accuracy, Precision , Recall and F1 score.

"""

import argparse
import numpy as np
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data,one_hot,DataLoader
from utils.metrics import evaluate_model as compute_metrics


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference on test set")

    # model path
    parser.add_argument('--model_path',type=str,default='src/best_model.npy',help="Path to save model weights(relative path)")

    # dataset
    parser.add_argument('-d','--dataset',type=str,default='mnist',choices=["mnist","fashion_mnist"],help="Dataset to evaluate on")

    # batch size
    parser.add_argument('-b','--batch_size',type=int,default=32,help="Batch size for inference")

    # architecture parameters
    parser.add_argument('-nhl','--num_layers',type=int,default=2,help="Number of hidden layers")

    parser.add_argument('-sz','--hidden_size',type=int,nargs='+',default=[128,64],help="Number of neurons in each hidden layer")

    parser.add_argument("-a","--activation",type=str,default="relu",choices=["sigmoid","tanh","relu","leaky_relu"],help="Activation function")

    # loss and optimizer
    parser.add_argument("-l","--loss",type=str,default="cross_entropy",choices=["mse","cross_entropy"],help="Loss function")

    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("-o", "--optimizer", type=str, default='adam', 
                    choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],
                    help="Optimization algorithm")
    parser.add_argument("-wd","--weight_decay",type=float,default=0.0,help="Weight decay")

    parser.add_argument("-wi","--weight_init",type=str,default="xavier",choices=["random","xavier","he","zeros"],help="Weight Initialization")

    parser.add_argument("-gc","--gradient_clip",type=float,default=0.0,help="Gradient clipping threshold")

    args = parser.parse_args()

    # ensure num_layer matches hidden_size length
    if len(args.hidden_size)!=args.num_layers:
        args.num_layers = len(args.hidden_size)
    
    # adding required attributes for model construction
    args.input_size=784
    args.output_size=10
    args.epochs=1
    args.wandb_project = "DA6401_Assignment_1"

    return args

def load_model(model_path,args):
    try:
        model = NeuralNetwork(args)

        # Load weights from .npy file
        weights_data = np.load(model_path,allow_pickle=True).item()
        model.set_weights(weights_data)

        print(f"Model Loaded Successfully from: {model_path}")
        return model
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}")
        print(f"Please ensure the model exists at the path: {model_path}")
        return None
    
    except Exception as e:
        print(f"Error: failed to load model:{str(e)}")
        return None
    

def evaluate_model(model,x_test,y_test,batch_size=32):
    # convert labels to one hot if not done
    if y_test.ndim==1:
        y_test_oh = one_hot(y_test,num_classes=10)
        y_test_labels = y_test
    else:
        y_test_oh=y_test
        y_test_labels = np.argmax(y_test,axis=1)
    
    # create data loader
    test_loader = DataLoader(x_test,y_test_oh,batch_size=batch_size,shuffle=False)

    all_logits=[]
    all_preds=[]
    all_losses=[]

    for x_batch,y_batch in test_loader:
        # forward pass
        logits = model.forward(x_batch)
        all_logits.append(logits)

        # compute loss
        loss  =  model.compute_loss(logits,y_batch)
        all_losses.append(loss)

        # get predictions
        preds = np.argmax(logits,axis=1)
        all_preds.extend(preds)

    # aggregate results
    logits_array = np.vstack(all_logits)
    preds_array = np.array(all_preds)
    avg_loss = float(np.mean(all_losses))

    # compute metrics
    metrics = compute_metrics(preds_array,y_test_labels,num_classes=10)

    # return dictionary 
    return {
        "logits":logits_array,
        "loss":avg_loss,
        "accuracy":metrics["accuracy"],
        "f1":metrics["f1"],
        "precision":metrics["precision"],
        "recall":metrics["recall"],
        "confusion_matrix":metrics["confusion_matrix"]
    }

def main():
    # parse arguments
    args = parse_arguments()
    print("\n"+"="*70)
    print("INFERENCE: Evaluating Trained Model")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Model path: {args.model_path}")
    print(f"Batch size:{args.batch_size}")
    print(f"Architecture:{args.num_layers} hidden layes {args.hidden_size}")
    print(f"Activation: {args.activation}")
    print("="*70+"\n")

    # load dataset
    print(f"Loading {args.dataset} dataset...")
    try:
        x_train,x_val,x_test,y_train,y_val,y_test = load_data(args.dataset)
        print(f"Dataset Loaded")
        print(f"Test set shape: {x_test.shape}\n")
    except Exception as e:
        print(f"Error: Failed to load dataset: {str(e)}")
        return None
    
    # load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, args)
    if model is None:
        return None
    print()
    
    # Evaluate model
    print("Running inference on test set...")
    results = evaluate_model(model, x_test, y_test, batch_size=args.batch_size)
    
    if results is None:
        return None
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Loss:      {results['loss']:.6f}")
    print(f"Accuracy:  {results['accuracy']:.6f}")
    print(f"Precision: {results['precision']:.6f}")
    print(f"Recall:    {results['recall']:.6f}")
    print(f"F1-Score:  {results['f1']:.6f}")
    print("="*70 + "\n")
    
    print("Evaluation complete!\n")
    
    return results


if __name__ == '__main__':
    main()