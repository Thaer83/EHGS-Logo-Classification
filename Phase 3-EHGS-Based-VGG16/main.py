# main.py
from data_preparation import load_and_preprocess_image
from optimizer.pso_optimizer import optimize_hyperparameters
from models.vgg16_model import construct_model
from trainers.model_trainer import train_model, test_model
import numpy as np

def main():
    # Data loading and preprocessing
    path = "flickr_logos_27_dataset_original/flickr_logos_27_dataset_images"
    annot = "flickr_logos_27_dataset_original/flickr_logos_27_dataset_training_set_annotation.txt"
    train_images, train_labels, train_labelsnum, train_labelsnumSet = load_and_preprocess_image(path, annot)
    
    # Convert labels to categorical
    train_labels = np.array(train_labelsnum)
    train_labels = to_categorical(train_labels, num_classes=len(train_labelsnumSet))
    
    # Split data into training and testing
    # Assuming you have a utility function for this or use train_test_split as before
    # train_x, test_x, train_y, test_y = ...

    # Optimize hyperparameters (This is where PSO optimization would be invoked)
    optimized_hyperparameters = optimize_hyperparameters(train_images, train_labels, train_labelsnumSet)
    
    # Construct, train, and evaluate the model using the optimized hyperparameters
    # model = construct_model(...)
    # history = train_model(...)
    # test_model(...)

if __name__ == "__main__":
    main()
