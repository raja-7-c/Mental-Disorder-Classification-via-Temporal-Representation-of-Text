import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_data(data_folder, label_csv_path, test_size=0.2, random_state= None):
    # Load labels from the label CSV file
    label_df = pd.read_csv(label_csv_path)

    # Create a mapping of subject IDs to labels
    label_dict = dict(zip(label_df['SubjectID'], label_df['Label']))

    # Initialize lists to store data and labels
    data = []
    labels = []

    # Loop through the CSV files in the data folder
    for filename in os.listdir(data_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_folder, filename)
            
            subject_id = os.path.splitext(filename)[0]  # Convert the matched string to an integer
            
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Assuming 'cosine similarity' is the column name containing time series data
            time_series = df['Cosine_Similarity'].values

            # Convert to a list
            time_series_list = time_series.tolist()

            # Append the data and corresponding label to the lists
            data.append(time_series_list)
            labels.append(label_dict.get(subject_id, 'unknown'))  # Handle cases where subject ID is not found in the label CSV

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Split the data into training and testing sets
    X_train, X_val, y_train, y_val = train_test_split(data, encoded_labels, test_size=test_size, random_state = random_state)

    # Find the maximum sequence length in your data
    max_sequence_length_train = max(len(seq) for seq in X_train)

    # Pad sequences to the maximum length
    padded_X_train = pad_sequences(X_train, maxlen=max_sequence_length_train, dtype='float32', padding="post", value=-1)
    padded_X_val = pad_sequences(X_val, maxlen=max_sequence_length_train, dtype='float32', padding="post", value=-1)

    # Convert to numpy arrays
    X_train = np.array(padded_X_train)
    X_val = np.array(padded_X_val)

    x_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    x_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

    return x_train, x_val, y_train, y_val, max_sequence_length_train

def permute_series(x_series):
    
    permuted_x = []
    for i in range(len(x_series)):
        permuted_x.append(np.random.permutation(x_series[i]))
        
    return np.array(permuted_x)

def load_and_slice_data(train_path, labels_path, test_path, feature_ranking_path, correlation_threshold): 
    # Load training data
    X = pd.read_csv(train_path, index_col=0)
    # # Permute the rows
    # X = permute_series(X)
    # Load labels
    labels_df = pd.read_csv(labels_path, header=None)
    Y = labels_df.iloc[:, :]
    Y = Y.set_index(0)
    Y.index.name = None

    y = Y[1]

    # Load test data
    X_Test = pd.read_csv(test_path, index_col=0)



    # Save the original index of X_Test
    test_index = X_Test.index

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2, random_state=92)

    # Train a RandomForestClassifier
    classifier = RandomForestClassifier(random_state=89)
    classifier.fit(X_train, y_train)

    desired_num_features = 30
    # Get feature importances
    importances = classifier.feature_importances_
    top_n_indices = importances.argsort()[-desired_num_features:][::-1]

    # Get top features column names
    top_n_column_names = X.columns[top_n_indices]

    # Calculate the correlation matrix for the top features
    correlation_matrix = X[top_n_column_names].corr()

    # Load feature rankings
    feature_rankings = pd.read_csv(feature_ranking_path)

    # Convert the Index to a DataFrame
    top_n_column_names_df = pd.DataFrame(index=top_n_column_names)

    # Find and print pairs with correlation greater than the threshold, sorted by correlation value
    printed_pairs = set()  # Use a set to store printed pairs
    sorted_pairs = []

    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            correlation_value = correlation_matrix.iloc[i, j]
            if abs(correlation_value) > correlation_threshold:
                pair = (correlation_matrix.columns[i], correlation_matrix.columns[j])
                reversed_pair = (pair[1], pair[0])

                # Check if the pair or its reversed version is already printed
                if pair not in printed_pairs and reversed_pair not in printed_pairs:
                    printed_pairs.add(pair)
                    sorted_pairs.append((pair, correlation_value))

    # Sort pairs by correlation value
    sorted_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    # Remove the lowest-ranking feature from each pair in top_n_column_names_df
    for pair, _ in sorted_pairs:
        feature1, feature2 = pair
        if feature1 in top_n_column_names_df.index and feature2 in top_n_column_names_df.index:
            ranking1 = feature_rankings.loc[feature_rankings['Feature'] == feature1, 'Ranking'].values
            ranking2 = feature_rankings.loc[feature_rankings['Feature'] == feature2, 'Ranking'].values

            if ranking1 > ranking2:
                top_n_column_names_df = top_n_column_names_df.drop([feature1])

            else:
                top_n_column_names_df = top_n_column_names_df.drop([feature2])

    # Extract the final column names from the DataFrame
    final_column_names = top_n_column_names_df.index

    index_positions_exp = [X.columns.get_loc(col) for col in final_column_names]

    X_train_top = X_train.iloc[:, index_positions_exp].to_numpy()
    X_val_top = X_val.iloc[:, index_positions_exp].to_numpy()
    X_test_top = X_Test.iloc[:, index_positions_exp].to_numpy()

    return X_train_top, y_train, X_val_top, y_val, X_test_top, test_index, len(final_column_names)