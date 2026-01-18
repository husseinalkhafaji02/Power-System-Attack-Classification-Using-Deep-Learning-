import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(folder_path, label_column, drop_columns=None):
    # Find all CSV files in the folder
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    # Read and concatenate all CSV files
    df_list = [pd.read_csv(f) for f in all_files]
    data = pd.concat(df_list, ignore_index=True)
    #data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

    # Drop unwanted columns if specified
    if drop_columns:
        data = data.drop(columns=drop_columns)


    # Balance the classes
    min_class_size = data[label_column].value_counts().min()
    balanced_data = (
        data.groupby(label_column, group_keys=False)
        .apply(lambda x: x.sample(min_class_size, random_state=42))
        .sample(frac=1, random_state=42)  # Shuffle again after balancing
        .reset_index(drop=True)
    )

   
    # Separate features and labels
    X = balanced_data.drop(columns=[label_column])
    y = balanced_data[label_column]
    
    # Map string labels: 'Natural' -> 0, 'Attack' -> 1
    y = y.map({'Natural': 0, 'Attack': 1})

    # Convert all columns to numeric, coerce errors to NaN
    X = X.apply(pd.to_numeric, errors='coerce')
   
    # Replace inf/-inf with NaN, then fill NaN with column mean (or zero)
    X = X.replace([float('inf'), float('-inf')], float('nan'))
    X = X.fillna(X.mean())
    
    

    # Normalize data (standardization: mean=0, std=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
   

    # Convert DataFrame to numpy array before reshaping
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Split the dataset into 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    



    # Print class counts after balancing
    print("Class counts after balancing:")
    print(balanced_data[label_column].value_counts())

    return X_train, X_test, y_train, y_test, X_val, y_val, scaler

if __name__ == "__main__":
    drop_columns = ['R2-PM8:V', 'R1-PM8:V', 'R2-PM9:V', 'R4-PA9:VH', 'R2-PA8:VH', 'R3-PA8:VH']  # Columns to drop

    x_train, x_test, y_train, y_test, x_val, y_val = preprocess_data(
        r"C:\Users\alkha\Downloads\binaryAllNaturalPlusNormalVsAttacks",
        'marker',
        drop_columns
    )
