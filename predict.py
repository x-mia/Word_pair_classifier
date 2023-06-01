#!/usr/bin/env python
# coding: utf-8

## Pre-trained model testing on new data
# Importing

import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


def NormalizeData(df, column_name):
    df[column_name] = df[column_name].replace('-', 0)
    df[column_name] = df[column_name].astype(float)
    column_log = np.log(df[column_name] + 0.001)
    scaled_column = list((column_log - np.min(column_log)) / (np.max(column_log) - np.min(column_log)))
    df[f"scaled_{column_name}"] = scaled_column
    return df


def process_test_data(test_df,src_lng, tgt_lng):
    test_df = NormalizeData(test_df, "position")
    test_df = NormalizeData(test_df, "freqs")
    test_df = NormalizeData(test_df, "index")
    test_data = []
    test_labels = []
    word_pairs = []
    for _, row in test_df.iterrows():
        src_w = row[src_lng]
        tgt_w = row[tgt_lng]
        word_pairs.append([src_w, tgt_w])
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
    return test_data, test_labels, test_df, word_pairs


def load_model(path):
    model = tf.keras.models.load_model(path)
    return model


def make_prediction(model, test_data, test_labels, test_df, word_pairs):        
    predicted_output_data = model.predict(test_data)
    binary_predictions = np.round(predicted_output_data)
    test_df['predictions'] = binary_predictions 

    for index, value in enumerate(binary_predictions):
        print("Word pair:", word_pairs[index], "Label: ", test_labels[index], "Prediction: ", value)


    tn, fp, fn, tp = confusion_matrix(test_labels, np.round(model.predict(test_data))).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    correct_predictions = np.sum(binary_predictions == test_labels)
    accuracy = correct_predictions / len(test_labels)

    f1 = f1_score(test_labels, binary_predictions)

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 score: ", f1)
    return test_df


def main(src_lng, tgt_lng, model_path, test_path, output):
    print("Loading test data.")
    test_df = pd.read_csv(test_path)
    print("Loading the model.")
    model = load_model(model_path)
    print("Processing test data.")
    test_data, test_labels, test_df, word_pairs = process_test_data(test_df, src_lng, tgt_lng)
    print("Making predictions...")
    test_df = make_prediction(model, test_data, test_labels, test_df, word_pairs)
    ("Saving dataframe.")
    test_df.to_csv(output, index=False)
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training and Evaluation")
    parser.add_argument("--src_lng", type=str, help="Language code of the source language")
    parser.add_argument("--tgt_lng", type=str, help="Language code of the target language")
    parser.add_argument("--model_path", type=str, help="Path to pre-trained model")
    parser.add_argument("--test_path", type=str, help="Path to the test dataframe file")
    parser.add_argument("--output", type=str, help="Path to save the model")

    args = parser.parse_args()

    main(args.src_lng, args.tgt_lng, args.model_path, args.test_path, args.output)

