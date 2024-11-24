import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications import InceptionV3 #(GoogleNet)
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import ResNet152



from keras import layers, models
#from tensorflow.keras.layers import Flatten, Dense
#from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import time
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import optimizers

def construct_model(train_ds, train_labelsnumSet,num_neurons=4096, learning_rate=0.001, activation='relu', optimizer_idx=0):
    print("--------------VGG16 no imagenet----------------")
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=train_ds[0].shape)
    base_model.trainable = False  # Freeze the base model (Not trainable weights)
    
    # Add new layers
    flatten_layer = layers.Flatten()
    dense_layer_1 = layers.Dense(num_neurons, activation=activation)
    dense_layer_2 = layers.Dense(num_neurons, activation=activation)
    prediction_layer = layers.Dense(len(train_labelsnumSet), activation='softmax')

    model = models.Sequential([
        base_model,
        flatten_layer,
        dense_layer_1,
        dense_layer_2,
        prediction_layer
    ])
    
    # Map optimizer_idx to an optimizer
    optimizers_list = [optimizers.Adam(learning_rate=learning_rate), 
                       optimizers.SGD(learning_rate=learning_rate, momentum=0.9), 
                       optimizers.RMSprop(learning_rate=learning_rate)]
    optimizer = optimizers_list[optimizer_idx % len(optimizers_list)]  # Cycle through list based on idx

    model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy', Precision(), Recall()])

    return model

def train_model(model, train_ds, train_labels, validation_split=0.2, epochs=20, batch_size=32):
    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5, restore_best_weights=True)
    history = model.fit(train_ds, train_labels, epochs=epochs, validation_split=validation_split, callbacks=[es], batch_size=batch_size)
    return history

def save_model(model, save_path='drive/MyDrive/Colab Notebooks/flickr_logos_27_dataset/my_model_final.h5'):
    model.save(save_path)

def load_model_for_testing(load_path='drive/MyDrive/Colab Notebooks/flickr_logos_27_dataset/my_model_final.h5'):
    model = load_model(load_path)
    return model

def test_model(model, test_ds, y_test):
    with tf.device('/cpu:0'):
        #Evaluate the model performance
        results = model.evaluate(test_ds, y_test, batch_size=128)
        # Start timing right before prediction
        start_time = time.time()
        # Get model predictions for the test dataset
        predictions = model.predict(test_ds)
        # Stop timing right after prediction
        end_time = time.time()
    # Calculate the inference time
    inference_time = end_time - start_time
    return results,predictions,inference_time