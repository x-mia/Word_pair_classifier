#!/usr/bin/env python
# coding: utf-8

# Importing

import argparse
import random
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def NormalizeData(df, column_name):
    df[column_name] = df[column_name].replace('-', 0)
    df[column_name] = df[column_name].astype(float)
    column_log = np.log(df[column_name] + 0.001)
    scaled_column = list((column_log - np.min(column_log)) / (np.max(column_log) - np.min(column_log)))
    df[f"scaled_{column_name}"] = scaled_column
    return df


def process_train_data(train_df, precision):
    if precision == 1:
        train_df = train_df[train_df['position'] == 0]
        train_df = train_df.rename(columns={'position': 'scaled_position'})
    if precision == 5:
        train_df = train_df[train_df['position'] < 5]
        train_df = NormalizeData(train_df, "position")
    if precision == 10:
        train_df = NormalizeData(train_df, "position")
    train_df = NormalizeData(train_df, "freqs")
    train_df = NormalizeData(train_df, "index")
    train_data = []
    train_labels = []
    for _, row in train_df.iterrows():
        data_x = [
            row['score'],
            row['rel_score'],
            row['ratio_score'],
            row['scaled_position'],
            row['scaled_index'],
            row['scaled_freqs']
        ]
        corr = row['correctness']
        if corr == 'yes':
            train_labels.extend([[1]] * 3)
            data_y = data_x[:]
            train_data.append(data_y)
            data_z = data_x[:]
            train_data.append(data_z)
            if precision == 5 or precision == 1:
                train_labels.append([1])
                data_w = data_x[:]
                train_data.append(data_w)
        elif corr == 'no':
            train_labels.append([0])
            if precision == 5 or precision == 1:
                train_labels.append([0])
                data_q = data_x[:]
                train_data.append(data_q)
          
        train_data.append(data_x)
    return train_data, train_labels


def process_test_data(test_df, precision):
    if precision == 1:
        test_df = test_df[test_df['position'] == 0]
        test_df = test_df.rename(columns={'position': 'scaled_position'})
        test_df['scaled_position'] = test_df['scaled_position'].replace('-', 0)
        test_df['scaled_position'] = test_df['scaled_position'].astype(float)
    if precision == 5:
        test_df = test_df[test_df['position'] < 5]
        test_df = NormalizeData(test_df, "position")
    if precision == 10:
        test_df = NormalizeData(test_df, "position")
    test_df = NormalizeData(test_df, "freqs")
    test_df = NormalizeData(test_df, "index")
    test_data = []
    test_labels = []
    for _, row in test_df.iterrows():
        data_x = [
            row['score'],
            row['rel_score'],
            row['ratio_score'],
            row['scaled_position'],
            row['scaled_index'],
            row['scaled_freqs']
        ]
        corr = row['correctness']
        if corr == 'yes':
            test_labels.append([1])
        elif corr == 'no':
            test_labels.append([0])
          
        test_data.append(data_x)
    return test_data, test_labels


def train_model(train_data, train_labels, test_data, test_labels):
    # Define the callback function
    class PrintLossCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Loss = {logs['loss']:.4f}")

    # Define the input and output dimensions
    input_dim = 6
    output_dim = 1

    # Define the input layer
    inputs = tf.keras.Input(shape=(input_dim,))
    hidden_layer_1 = tf.keras.layers.Dense(24, activation='tanh')(inputs)
    hidden_layer_2 = tf.keras.layers.Dense(12, activation='tanh')(hidden_layer_1)
    hidden_layer_3 = tf.keras.layers.Dense(8, activation='tanh')(hidden_layer_2)
    output_layer = tf.keras.layers.Dense(output_dim, activation='sigmoid')(hidden_layer_3)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=output_layer)

    # Compile the model with the binary cross-entropy loss function and the Adam optimizer
    model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

    # Train the model
    history = model.fit(train_data, train_labels, epochs=500, verbose=0, callbacks=[PrintLossCallback()],
                        shuffle=True, validation_data=(test_data, test_labels))

    return model, history


def evaluate_model(model, train_data, train_labels, test_data, test_labels):
    # Evaluate the model on the training data
    loss_train, accuracy_train = model.evaluate(train_data, train_labels, verbose=0)
    print("Training loss:", loss_train)
    print("Training accuracy:", accuracy_train)

    # Evaluate the model on the test data
    loss_test, accuracy_test = model.evaluate(test_data, test_labels, verbose=0)
    print("Testing loss:", loss_test)
    print("Testing accuracy:", accuracy_test)
    
    
def plot_model_history(history):
    # Plot the training and validation accuracy over epochs
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("Accuracy.png", dpi=150)
    plt.show()

    # Plot the training and validation loss over epochs
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("Loss.png", dpi=150)
    plt.show()
    
    
def main(precision, train_path, test_path, plot_fig, output):
    print("Loading train data")
    train_df = pd.read_csv(train_path)
    print("Processing train data")
    train_data, train_labels = process_train_data(train_df, precision)
    print("Loading test data")
    test_df = pd.read_csv(test_path)
    print("Processing test data")
    test_data, test_labels = process_test_data(test_df, precision)
    print("Training")
    model, history = train_model(train_data, train_labels, test_data, test_labels)
    print("Evaluating")
    evaluate_model(model, train_data, train_labels, test_data, test_labels)
    if plot_fig == True:
        print("Plotting")
        plot_model_history(history)
    print("Saving")
    model.save(output)
    print("Done")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training and Evaluation")
    parser.add_argument("--precision", type=int, choices=[1, 5, 10], help="Select precision: 1, 5, or 10")
    parser.add_argument("--train_path", type=str, help="Path to the train dataframe file")
    parser.add_argument("--test_path", type=str, help="Path to the test dataframe file")
    parser.add_argument("--plot_fig", type=bool, help="True/False if you want to plot Accuracy/Loss to Epochs")
    parser.add_argument("--output", type=str, help="Path to save the model")

    args = parser.parse_args()

    main(args.precision, args.train_path, args.test_path, args.plot_fig, args.output)

