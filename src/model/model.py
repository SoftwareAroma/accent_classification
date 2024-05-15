# !/usr/bin/env python
"""
    The main model implementation is contained in this file
    the configurations may be outside the file but the main
    classes and functions are contained in this file. 
"""
import os
import time
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from keras.applications import VGG16

from constants import EPOCHS, LEARNING_RATE, OUTPUT_DIR
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_ENABLE_ONEDNN_OPTS']= "0"

# if gpu is available, use it
if tf.config.list_physical_devices('GPU'):
    device = 'gpu'
else:
    device = 'cpu'


class MainModel:
    """
        The main model class that contains all the functions
        and methods that are used to train, test, evaluate and
        save the model.
        :param dataset_dir: Path to the directory containing the dataset.
        :param labels: List of labels in the dataset.
        :param mode: Mode of the model, either train, test or evaluate.
        :param model_type: Type of model to use, either cnn or rnn.
        :param model_name: Name of the model to save.
        :param model_path: Path to save the model.
        :param epochs: Number of epochs to train the model.
        :param learning_rate: Learning rate of the model.
        :param batch_size: Batch size of the model.
        :param num_workers: Number of workers to use.
        :param verbose: Whether to print out the progress.
        :param **kwargs: Additional keyword arguments.
        :return: A dictionary containing the accuracy, loss, f1_score, confusion_matrix and time_taken.
    """
    def __init__(
            self,
            x_train: any = None,
            y_train: any = None,
            x_validation: any = None,
            y_validation: any = None,
            mode: str = 'train',
            model_name: str = 'model.keras',
            model_dir: str = OUTPUT_DIR,
            epochs: int = EPOCHS,
            batch_size: int = 32,  # 16, 32, 64, 128
            learning_rate: float = LEARNING_RATE,
            verbose: int = 1,
            **kwargs: dict[str, any]
    ) -> None:
        # parameters
        if x_train is None or y_train is None or x_validation is None or y_validation is None:
            raise ValueError(
                "x_train, y_train, x_validation and y_validation cannot be None"
            )
        self.x_train = x_train
        self.y_train = y_train
        self.x_validation = x_validation
        self.y_validation = y_validation
        self.mode = mode
        self.model_name = model_name
        self.model_dir = model_dir
        self.model_path = f"{model_dir}/{model_name}"
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose
        self.kwargs = kwargs

        # model
        self.model = None
        self.accuracy = None
        self.loss = None
        self.f1_score = None
        self.confusion_matrix = None
        self.time_taken = None
        self.predictions = []


    def check_path(self) -> None:
        """
            Checks if the model path exists.
        """
        # check if the model directory exists
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def create_cnn_model(self) -> object:
        """
            Creates the model.
            :return: The model.
        """
        # Get row, column, and class sizes
        rows = self.x_train[0].shape[0]
        cols = self.x_train[0].shape[1]
        val_rows = self.x_validation[0].shape[0]
        val_cols = self.x_validation[0].shape[1]
        num_classes = len(self.y_train[0])

        # input image dimensions to feed into 2D ConvNet Input layer
        input_shape = (rows, cols, 1)
        self.x_train = self.x_train.reshape(self.x_train.shape[0], rows, cols, 1)
        self.x_validation = self.x_validation.reshape(
            self.x_validation.shape[0], 
            val_rows, val_cols, 1
        )

        # model construction
        self.model = keras.models.Sequential(name="MainModel")
        self.model.add(
            keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation=keras.activations.relu,  # 'relu'
                data_format="channels_last",
                input_shape=input_shape,
            )
        )
        self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation=keras.activations.relu))  # 'relu'
        self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(keras.layers.Dropout(0.25))

        # Flattening the 2D arrays for fully connected layers
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(128, activation=keras.activations.relu))  # 'relu'
        # self.model.add(tf.keras.layers.Dropout(0.5))
        # Output Layer
        self.model.add(
            keras.layers.Dense(
                num_classes,  
                activation=keras.activations.softmax
            )
        )
        # compile model
        self.model.compile(
            loss=keras.losses.CategoricalCrossentropy(from_logits=False),
            optimizer=keras.optimizers.Adadelta(learning_rate=self.learning_rate),
            metrics=['accuracy']
        )
        return self.model
    
    
    def create_transformer_model(self) -> object:
        """
            Creates the Transformer model.
            :return: The model.
        """
        num_classes = len(self.y_train[0])
        # Get row, column, and class sizes
        rows = self.x_train[0].shape[0]
        cols = self.x_train[0].shape[1]
        # input image dimensions to feed into 2D ConvNet Input layer
        input_shape = (rows, cols, 1)
        
        # Define the encoder input
        encoder_inputs = keras.layers.Input(shape=input_shape, name='encoder_input')
        
        # Define the decoder input
        decoder_inputs = keras.layers.Input(shape=input_shape, name='decoder_input')
        
        # Define the encoder and decoder outputs (placeholders for now)
        encoder_outputs = keras.layers.Dense(64)(encoder_inputs)
        decoder_outputs = keras.layers.Dense(64)(decoder_inputs)
        
        # Define the multi-head attention layer
        attention_layer = keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)
        
        # Apply the attention layer to the encoder and decoder outputs
        encoder_attention = attention_layer(query=encoder_outputs, key=encoder_outputs, value=encoder_outputs)
        decoder_attention = attention_layer(query=decoder_outputs, key=decoder_outputs, value=encoder_attention)
        
        # Flatten the merged output for dense layers
        flatten_output = keras.layers.Flatten()(decoder_attention)
        
        # Dense layers for classification
        dense_layer = keras.layers.Dense(128, activation=keras.activations.relu)(flatten_output)
        dropout_layer = keras.layers.Dropout(0.5)(dense_layer)
        output_layer = keras.layers.Dense(num_classes, activation=keras.activations.softmax)(dropout_layer)
        
        # Define the model
        self.model = keras.models.Model(
            inputs=[encoder_inputs, decoder_inputs],
            outputs=output_layer,
            name='TransformerModel'
        )
        
        # Compile the model
        self.model.compile(
            loss=keras.losses.BinaryCrossentropy(from_logits=False),
            optimizer=keras.optimizers.Adadelta(learning_rate=self.learning_rate),
            metrics=['accuracy']
        )
        return self.model

    def train_cnn_model(self) -> None:
        """
            Trains the model.
            :return: A dictionary containing the accuracy, loss, f1_score, confusion_matrix and time_taken.
        """
        # Image shifting
        datagen = keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.05)
        # start a timer to track the training time
        start_time = time.time()
        # Fit model using ImageDataGenerator
        self.model.fit(
            datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size),
            epochs=self.epochs,
            validation_data=(
                self.x_validation,
                self.y_validation
            ),
            verbose=self.verbose
        )
        # keep track of the training time
        self.time_taken = time.time() - start_time
        self.accuracy = self.model.history.history['accuracy']
        self.loss = self.model.history.history['loss']


    def train_transformer_model(self) -> None:
        """
            Trains the Transformer model.
        """
        # start a timer to track the training time
        start_time = time.time()
        # Fit the model
        self.model.fit(
            [self.x_train, self.x_train],  # encoder and decoder inputs
            self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=([self.x_validation, self.x_validation], self.y_validation),
            verbose=self.verbose
        )
        # keep track of the training time
        self.time_taken = time.time() - start_time
        self.accuracy = self.model.history.history['accuracy']
        self.loss = self.model.history.history['loss']


    def model_summary(self) -> None:
        """
            Prints the model summary.
        """
        # print the model summary
        print(self.model.summary())


    def evaluate_model(self, x_test: any, y_test: any) -> object:
        """
            Tests the model.
            :param x_test: The test data.
            :param y_test: The test labels.
            :return: A dictionary containing the accuracy, loss, f1_score, confusion_matrix and time_taken.
        """
        # perform the testing y_test,
        self.loss, self.accuracy = self.model.evaluate(x_test, verbose=self.verbose)
        # get confusion matrix
        self.get_confusion_matrix(x_test, y_test)
        self.plot_confusion_matrix()
        return {
            "accuracy": self.accuracy,
            "loss": self.loss,
        }


    def plot_confusion_matrix(self) -> None:
        """
            Plots the confusion matrix.
        """
        # check if self.model_dir exists
        # self.check_path()
        
        # Get the confusion matrix
        if self.confusion_matrix is None:
            raise ValueError("Confusion matrix not computed. Please evaluate the model first.")
        
        # Plot the confusion matrix using seaborn heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            self.confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues"
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig('../../output/confusion_matrix.png')
        plt.show()


    def plot_model_stats(self) -> None:
        """
            Plots the training statistics.
        """
        # check if self.model_dir exists
        self.check_path()
        # plot the model statistics with matplotlib
        plt.figure(figsize=(10, 10))
        plt.plot(self.model.history.history['accuracy'])
        plt.plot(self.model.history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(f'{self.model_dir}/model_accuracy.png')

    def plot_model(self) -> None:
        """
            Plots the model.
        """
        # check if self.model_dir exists
        self.check_path()
        tf.keras.utils.plot_model(
            self.model,
            to_file=f'{self.model_dir}/model.png',
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=True,
            show_layer_activations=True,
            show_trainable=True,
            dpi=200
        )


    def save_model(self, path=None) -> None:
        """
            Saves the model.
        """
        # check if self.model_dir exists
        self.check_path()
        if path is None:
            # save and return the model
            self.model.save(filepath=f"{self.model_dir}/model5.keras")
        else:
            self.model.save(filepath=path)


    def predict_class_audio(self, mfcc_array:any) -> any:
        """
            Predict class based on MFCC samples

            :param mfcc_array: Numpy array of MFCCs
            :return: Predicted class of MFCC segment group
        """
        arr_mf = mfcc_array.reshape(
            mfcc_array.shape[0],
            mfcc_array.shape[1],
            mfcc_array.shape[2],
            1
        )
        y_predicted = self.model.predict_classes(arr_mf, verbose=0)
        return Counter(list(y_predicted)).most_common(1)[0][0]

    def predict_prob_class_audio(self, mfcc_array) -> any:
        """
            Predict class based on MFCC samples' probabilities
            :param mfcc_array: Numpy array of MFCCs
            :return: Predicted class of MFCC segment group
        """
        mf_arr = mfcc_array.reshape(
            mfcc_array.shape[0],
            mfcc_array.shape[1],
            mfcc_array.shape[2],
            1
        )
        y_predicted = self.model.predict_proba(mf_arr, verbose=0)
        return np.argmax(np.sum(y_predicted, axis=0))


    def predict_class_all(self, x_train: any) -> any:
        """
            :param x_train: List of segmented mfcc
            :return: list of predictions
        """
        for mfcc in x_train:
            self.predictions.append(self.predict_class_audio(mfcc))
        return self.predictions


    def get_confusion_matrix(self, x_test, y_test) -> any:
        """
            Create confusion matrix
            :param x_test: Test data
            :param y_test: Test labels
            :return: Confusion matrix
        """
        # If model is none then raise an error
        if self.model is None:
            raise ValueError("Model is None. Please train or load a model.")
        y_predicted = self.model.predict(x_test)
        y_predicted_classes = tf.argmax(y_predicted, axis=1).numpy()
        y_true_classes = tf.argmax(y_test, axis=1).numpy()
        # Compute the confusion matrix
        confusion_matrix = confusion_matrix(y_true_classes, y_predicted_classes)
        # plot and save confusion matrix 
        self.confusion_matrix = confusion_matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues"
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig('../../output/confusion_matrix.png')
        return confusion_matrix


    def get_accuracy(self, x_test, y_test) -> float:
        """
            Calculate accuracy of the model.
            :param x_test: Test data
            :param y_test: Test labels
            :return: Accuracy
        """
        if self.model is None:
            raise ValueError("Model is None. Please train or load a model.")
        self.get_confusion_matrix(x_test, y_test)
        accuracy = np.sum(np.diag(self.confusion_matrix)) / np.sum(self.confusion_matrix)
        return accuracy
    from sklearn.metrics import f1_score


    def calculate_f1_score(self, x_test, y_test) -> float:
        """
            Calculate the F1 score of the model.
            :param x_test: Test data
            :param y_test: Test labels
            :return: F1 score
        """
        # If model is none then raise an error
        if self.model is None:
            raise ValueError("Model is None. Please train or load a model.")
        y_predicted = self.model.predict(x_test)
        y_predicted_classes = tf.argmax(y_predicted, axis=1).numpy()
        y_true_classes = tf.argmax(y_test, axis=1).numpy()
        # Calculate the F1 score
        f1 = f1_score(y_true_classes, y_predicted_classes, average='weighted')
        return f1


    def info(self) -> dict:
        """
            Returns the information of the model.
        """
        if self.model is None:
            return {
                "model_name": self.model_name,
                "model_path": self.model_path,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "device": self.device,
                "verbose": self.verbose,
                "accuracy": self.accuracy,
                "loss": self.loss,
                "f1_score": self.f1_score,
                "confusion_matrix": self.confusion_matrix,
                "time_taken": self.time_taken,
                "predictions": self.predictions,
                "kwargs": self.kwargs,
            }
        else:
            return self.model.summary()

    def word_error_rate(self, y_predicted, y_test) -> any:
        # calculate word error rate
        pass
        

    def __str__(self) -> str:
        """
            Returns the string representation of the model.
        """
        return str(self.info())
