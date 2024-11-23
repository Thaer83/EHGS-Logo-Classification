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

def construct_model(train_ds, train_labelsnumSet):
    print("--------------VGG16 no imagenet----------------")
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=train_ds[0].shape)
    #base_model = ResNet50V2(weights="imagenet", include_top=False, input_shape=train_ds[0].shape)
    #base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=train_ds[0].shape)
    #base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=train_ds[0].shape)
    #base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=train_ds[0].shape)
    #base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=train_ds[0].shape)
    #base_model = ResNet152(weights="imagenet", include_top=False, input_shape=train_ds[0].shape)
    #base_model = VGG16(weights=None, include_top=False, input_shape=train_ds[0].shape)
    base_model.trainable = False  # Freeze the base model (Not trainable weights)

    flatten_layer = layers.Flatten()
    dense_layer_1 = layers.Dense(4096, activation='relu')
    dense_layer_2 = layers.Dense(4096, activation='relu')
    prediction_layer = layers.Dense(len(train_labelsnumSet), activation='softmax')

    model = models.Sequential([
        base_model,
        flatten_layer,
        dense_layer_1,
        dense_layer_2,
        prediction_layer
    ])
    from tensorflow.keras.metrics import Precision, Recall

    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', Precision(), Recall()])
    '''model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy', 
        metrics=['accuracy', 'precision', 'recall']
        #metrics=['accuracy']
        )'''
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