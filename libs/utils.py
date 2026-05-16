import numpy as np
import os
import pandas as pd
import joblib
from tensorflow import keras
from sklearn.metrics import f1_score, classification_report
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from models.ml import anorexia_random_forest

 
def find_best_threshold(x_val, y_val, model_path, positive_class = 1):
    # Load the best saved model
    best_model = keras.models.load_model(model_path)

    # Make predictions on the validation data using the best model
    y_pred_prob = best_model.predict(x_val)

    # Find the optimal threshold
    best_threshold = None
    best_f1 = 0.0

    for threshold in np.arange(0.1, 1.0, 0.01):
        # Convert y_val to binary labels (0 or 1)
        y_val_binary = (y_val == positive_class).astype(int)

        # Use the threshold to make binary predictions
        y_pred_binary = (y_pred_prob[:, positive_class] > threshold).astype(int)

        f1 = f1_score(y_val_binary, y_pred_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold

def train_model(x_train, y_train, x_val, y_val, model, epochs=None, batch_size=None, model_checkpoint=None, class_weight = None):
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_checkpoint, save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.01
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]
    model.compile(
        optimizer="Adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(x_val, y_val),
        class_weight=class_weight,
        verbose=1,
    )


def get_classification_report(test_folder, test_label_path, model_path, best_threshold, max_sequence_length_train, output_file_path, target_names= ['Control', 'Condition']):
    # Load the best model
    best_model = keras.models.load_model(model_path)

    # Initialize lists to store data and subject IDs
    data = []
    subject_ids = []

    # Loop through the CSV files in the test folder
    for filename in os.listdir(test_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(test_folder, filename)

            # Extract the subject ID from the filename (assuming the filename is in the format "subject_id.csv")
            subject_id = os.path.splitext(filename)[0]

            # Read the CSV file
            df = pd.read_csv(file_path)

            # Assuming 'cosine similarity' is the column name containing time series data
            time_series = df['Cosine_Similarity'].values

            # Convert to a list
            time_series_list = time_series.tolist()

            # Append the data and corresponding subject ID to the lists
            data.append(time_series_list)
            subject_ids.append(subject_id)

    # Convert data to numpy array
    X_test = data

    # Pad sequences to the maximum length
    X_test = pad_sequences(X_test, maxlen=max_sequence_length_train, dtype='float32', padding='post', truncating='post', value=-1)

    # Reshape the data for prediction
    x_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Make predictions on the test data using the best model
    y_pred_prob = best_model.predict(x_test)

    # Use the best threshold to make binary predictions
    y_pred_binary = (y_pred_prob[:, 1] > best_threshold).astype(int)

    # Assuming you have an array subject_ids containing subject IDs and an array y_pred_binary containing predictions

    # Define the path to the output text file
    output_file_path = output_file_path

    # Open the file for writing
    with open(output_file_path, 'w') as file:
        # Iterate through subject IDs and predictions and write them to the file
        for subject_id, prediction in zip(subject_ids, y_pred_binary):
            # Convert the prediction to a string
            prediction_str = str(prediction)

            # Write the SubjectID and Prediction with a space separator
            file.write(f'{subject_id} {prediction_str}\n')

    # Read the ground-truth file into a dictionary
    ground_truth = {}
    with open(test_label_path, 'r') as file:
        for line in file:
            subject, label = line.strip().split()
            ground_truth[subject] = int(label)

    # Read the predictions file into a dictionary
    predictions = {}
    with open(output_file_path, 'r') as file:
        for line in file:
            subject, label = line.strip().split()
            predictions[subject] = int(label)

    # Create lists to store true labels and predicted labels in the same order
    true_labels = []
    predicted_labels = []

    for subject in ground_truth:
        true_labels.append(ground_truth[subject])
        predicted_labels.append(predictions.get(subject, 0))  # Use 0 as the default if subject is not in predictions

    # Calculate the classification report
    class_report = classification_report(true_labels, predicted_labels, target_names=target_names)

    return class_report



def train_and_find_best_threshold(X_train_top, y_train, X_val_top, y_val, epochs, batch_size, model, model_path, learning_rate):

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    # Define callbacks
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=40, verbose=1, restore_best_weights=True)

    # Train the model with callbacks
    history = model.fit(X_train_top, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val_top, y_val), callbacks=[checkpoint, early_stopping])

    # Load the best saved model
    best_model = keras.models.load_model(model_path)

    # Evaluate the best model on the validation set
    y_pred_prob = best_model.predict(X_val_top)

    # Find the optimal threshold
    best_threshold = None
    best_f1 = 0.0

    for threshold in np.arange(0.1, 1.0, 0.01):
        y_pred_binary = (y_pred_prob > threshold).astype(int)
        f1 = f1_score(y_val, y_pred_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold



def find_best_threshold_ml(X_train_top, y_train, X_val_top, y_val, model, model_path):

    # Create a new Random Forest Classifier
    classifier_top = model

    classifier_top.fit(X_train_top, y_train)
    joblib.dump(classifier_top, model_path)

    best_model = joblib.load(model_path)

    # Evaluate the best model on the validation set
    y_pred_prob = best_model.predict(X_val_top)

    # Find the optimal threshold
    best_threshold = None
    best_f1 = 0.0

    for threshold in np.arange(0.1, 1.0, 0.01):
        y_pred_binary = (y_pred_prob > threshold).astype(int)
        f1 = f1_score(y_val, y_pred_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold





def print_classification_report(X_test_top, model_path, output_file_path, best_threshold, labels_path, test_index):
    # Load the best saved model
    best_model = keras.models.load_model(model_path)

    # Make predictions on the test data
    y_pred_prob = best_model.predict(X_test_top)

    # Use the best threshold to make binary predictions
    y_pred_binary = (y_pred_prob > best_threshold).astype(int)

    # Assuming y_pred_binary is a 2D array
    flat_y_pred = np.ravel(y_pred_binary)

    # Create a new DataFrame with only the 'Predicted_Label' column and no index
    predictions_df = pd.DataFrame({'Index': test_index, 'Predicted_Label': flat_y_pred})

    # Convert the 'Predicted_Label' column to integer (0 or 1)
    predictions_df['Predicted_Label'] = predictions_df['Predicted_Label'].astype(int)

    # Save the DataFrame to a text file without the index and with 0 or 1 as values
    predictions_df.to_csv(output_file_path, sep='\t', header=False, index=False)

    # Read the ground-truth file into a dictionary
    ground_truth = {}
    with open(labels_path, 'r') as file:
        for line in file:
            subject, label = line.strip().split()  # Split subject and label
            ground_truth[subject] = int(label)

    # Read the predictions file into a dictionary
    predictions = {}
    with open(output_file_path, 'r') as file:
        for line in file:
            subject, label = line.strip().split()
            predictions[subject] = int(label)

    # Create lists to store true labels and predicted labels in the same order
    true_labels = []
    predicted_labels = []

    for subject in ground_truth:
        true_labels.append(ground_truth[subject])
        predicted_labels.append(predictions.get(subject, 0))  # Use 0 as the default if subject is not in predictions

    # Calculate the classification report
    target_names = ['Control', 'Condition']  # Labels for binary classification
    class_report = classification_report(true_labels, predicted_labels, target_names=target_names)
    
    return class_report




def print_classification_report_ml(X_test_top, model_path, output_file_path, best_threshold, labels_path, test_index):
    # Load the best saved model
    best_model = joblib.load(model_path)

    # Make predictions on the test data
    y_pred_prob = best_model.predict(X_test_top)

    # Use the best threshold to make binary predictions
    y_pred_binary = (y_pred_prob > best_threshold).astype(int)

    # Assuming y_pred_binary is a 2D array
    flat_y_pred = np.ravel(y_pred_binary)

    # Create a new DataFrame with only the 'Predicted_Label' column and no index
    predictions_df = pd.DataFrame({'Index': test_index, 'Predicted_Label': flat_y_pred})

    # Convert the 'Predicted_Label' column to integer (0 or 1)
    predictions_df['Predicted_Label'] = predictions_df['Predicted_Label'].astype(int)

    # Save the DataFrame to a text file without the index and with 0 or 1 as values
    predictions_df.to_csv(output_file_path, sep='\t', header=False, index=False)

    # Read the ground-truth file into a dictionary
    ground_truth = {}
    with open(labels_path, 'r') as file:
        for line in file:
            subject, label = line.strip().split()  # Split subject and label
            ground_truth[subject] = int(label)

    # Read the predictions file into a dictionary
    predictions = {}
    with open(output_file_path, 'r') as file:
        for line in file:
            subject, label = line.strip().split()
            predictions[subject] = int(label)

    # Create lists to store true labels and predicted labels in the same order
    true_labels = []
    predicted_labels = []

    for subject in ground_truth:
        true_labels.append(ground_truth[subject])
        predicted_labels.append(predictions.get(subject, 0))  # Use 0 as the default if subject is not in predictions

    # Calculate the classification report
    target_names = ['Control', 'Condition']  # Labels for binary classification
    class_report = classification_report(true_labels, predicted_labels, target_names=target_names)
    
    return class_report