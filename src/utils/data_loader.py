"""
This file contains Data Loading utilities for MNIST and Fashion-MNIST.
It contains Data loading and preprocessing methods to apply before sending the data for training.
"""

import numpy as np
from tensorflow.keras.datasets import mnist,fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(dataset_name="mnist",val_size=0.15,random_state=42):
    # loading  the dataset as per dataset name
    print(f"Loading dataset {dataset_name}")
    if dataset_name=="mnist":
        (x_train,y_train),(x_test,y_test) = mnist.load_data()
    elif dataset_name=="fashion_mnist":
        (x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Enter valid dataset name: (mnist or fashion_mnist)")
    
    # flatten images (images shape: (N,28,28))
    x_train = x_train.reshape(-1,784).astype(np.float32)
    x_test = x_test.reshape(-1,784).astype(np.float32)

    # normalize theintensity values to [0,1]
    x_train = x_train/255.0
    x_test = x_test/255.0

    #split the x_train into training and validation set
    X_train,X_val,Y_train,Y_val = train_test_split(x_train,y_train,test_size=val_size,random_state=random_state,stratify=y_train)
    # stratify is used to maintain class distribution 

    print(f"Dataset shape:")
    print(f"X_train: {X_train.shape},Y_train: {Y_train.shape}")
    print(f"X_val: {X_val.shape},y_val:{Y_val.shape}")
    print(f"X_test: {x_test.shape},y_test:{y_test.shape}")
    return X_train,X_val,x_test,Y_train,Y_val,y_test

def one_hot(target_column,num_classes=10):
    return np.eye(num_classes)[target_column]

class DataLoader:
    """
    Dataloader to create min-batches for training neural networks.

    """
    def __init__(self,X,Y,batch_size=32,shuffle=True,random_state=42):
        self.X = X
        self.Y=Y
        self.batch_size=batch_size
        self.shuffle = shuffle
        self.random_state=random_state

        self.num_samples = X.shape[0]
        self.num_batches = int(np.ceil(self.num_samples/batch_size))
        self.indices = np.arange(self.num_samples)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_batch=0
        return self
    def __next__(self):
        # if all batches are processed, then Stop iteration
        if self.current_batch>=self.num_batches:
            raise StopIteration
        # Finding index where to start
        start = self.current_batch*self.batch_size
        # Finding index where to end
        end = min(start+self.batch_size,self.num_samples)
        # Extracting start to end indices
        batch_indices = self.indices[start:end]
        X_batch = self.X[batch_indices]
        Y_batch = self.Y[batch_indices]
        self.current_batch+=1
        return X_batch,Y_batch

def get_sample_images(input,target,samples_per_class=5,num_classes=10):
    sample_images={}
    sample_labels={}
    for class_label in range(num_classes):
        class_indices = np.where(target==class_label)[0]
        selected_indices = np.random.choice(class_indices,size=samples_per_class,replace=False)
        images = input[selected_indices].reshape(-1,28,28)
        labels = target[selected_indices]
        sample_images[class_label]=images
        sample_labels[class_label]=labels
    return sample_images,sample_labels

def get_class_distribution(target,num_classes=10):
    distribution = {}
    total_size = len(target)

    for class_label in range(num_classes):
        count = np.sum(target==class_label)
        distribution[class_label]=count
    percentages = {class_label: (count/total_size)*100 for class_label,count in distribution.items()}
    return distribution,percentages

def standardize_data(x_train,x_val,x_test):
    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train)
    x_val_std = scaler.transform(x_val)
    x_test_std = scaler.transform(x_test
                                  )
    return x_train_std,x_val_std,x_test_std
