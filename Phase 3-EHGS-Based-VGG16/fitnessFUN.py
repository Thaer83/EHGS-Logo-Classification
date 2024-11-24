# -*- coding: utf-8 -*-
"""
Created on Tue Mar 1 12:46:20 2024

@author: Thaer Thaher 
"""

import numpy as np
#from models.vgg16_model import construct_model
#from trainers.model_trainer import train_model
from tensorflow.keras.callbacks import EarlyStopping

from model_training import (
    construct_model,
    train_model,
    save_model,
    load_model_for_testing,
    test_model
)
          
#____________________________________________________________________________________       
def objective_function(hyperparameters, train_ds, train_labels, train_labelsnumSet):
    # This function should construct the model with hyperparameters,
    # train it, and return the optimization objective (e.g., -val_accuracy).
    # Unpack hyperparameters
    learning_rate, num_neurons, activation_idx, optimizer_idx, patience, batch_size = hyperparameters
    # Map indices to categorical values
    activation_functions = ['relu', 'sigmoid', 'tanh']
    optimizers = ['adam', 'sgd', 'rmsprop']
    activation_function = activation_functions[int(activation_idx) % len(activation_functions)]
    optimizer_type = optimizers[int(optimizer_idx) % len(optimizers)]

    # Round patience and batch_size to nearest integer (and ensure batch_size is a power of 2)
    patience = int(round(patience))
    batch_size = 2**int(round(np.log2(batch_size)))

    # Construct and compile model with these hyperparameters

    model = construct_model(train_ds, train_labelsnumSet, num_neurons=num_neurons, learning_rate=learning_rate,
                            activation_function=activation_function, optimizer_idx=optimizer_idx)
    
    # Modify early stopping to use the rounded patience
    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=patience, restore_best_weights=True)

    # Train model using the rounded batch_size
    history = model.fit(train_ds, train_labels, epochs=5, validation_split=0.2, callbacks=[es], batch_size=batch_size, verbose=0)

    # Objective to minimize
    val_accuracy = max(history.history['val_accuracy'])
    return -val_accuracy
#_____________________________________________________________________       

def optimize_hyperparameters(train_ds, train_labels, train_labelsnumSet):
    # Define lower and upper bounds for each hyperparameter
    lb = [0.0001, 128, 0, 0, 3, 16]  # Lower bounds
    ub = [0.01, 4096, 2, 2, 10, 128] # Upper bounds   or  ub = [0.01, 4096, 2, 2, 10, 128]
    
    dim = len(lb)
    # Call HGS
    optimal_hyperparameters, _ = HGS(objective_function, lb, ub, dim, 10, 100, args=(train_ds, train_labels, train_labelsnumSet))

    
    # Return the optimized hyperparameters

def getFunctionDetails(a):
    
    # [name, lb, ub, dim]
    param = {  0:["FN1",-1,1]

            }
    return param.get(a, "nothing")



