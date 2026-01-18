from preprocess import preprocess_data
from networkmodel import build_model_tuner
from sklearn.preprocessing import StandardScaler
from keras_tuner.engine.hyperparameters import HyperParameters
import pandas as pd

# Setup
hp = HyperParameters()
features = 122
classes = 2
comms_round = 100
lr = 0.0004571155244
loss = 'binary_crossentropy'
metrics = ['accuracy']
optimizer = 'adam'
num_customers = 5
epochs = 1 # safer for testing

# Preprocess shared test data
drop_columns = ['R2-PM8:V', 'R1-PM8:V', 'R2-PM9:V', 'R4-PA9:VH', 'R2-PA8:VH', 'R3-PA8:VH']
x_train, x_test, y_train, y_test, x_val, y_val, scaler = preprocess_data(
    "/work/projects/reu-2025/c-halkhafaji/cyber security detection model/binaryAllNaturalPlusNormalVsAttacks1",
    'marker',
    drop_columns
)

# Initialize global model
global_model = build_model_tuner(hp, input_dim=x_train.shape[1])
#global_model = Simple_global.build(features, classes)
global_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# Federated Training
for comm_round in range(comms_round):
    print(f"\nCommunication round {comm_round + 1}/{comms_round}")

    scaled_local_weight_list = [0 for _ in global_model.get_weights()]
    global_weights = global_model.get_weights()

    for customer in range(1, num_customers + 1):
        data = pd.read_csv(
            f'/work/projects/reu-2025/c-halkhafaji/cyber security detection model/binaryAllNaturalPlusNormalVsAttacks1/data{customer}.csv'
        )

        X_train0 = data.drop(columns=['marker'] + drop_columns)
        y_train0 = data['marker']
        y_train0 = y_train0.map({'Natural': 0, 'Attack': 1})  # convert labels to 0/1
        ytr = y_train0.values.astype('float32')
        ytr = y_train0.values
        # Convert to numeric, coerce errors to NaN
        X_train0 = X_train0.apply(pd.to_numeric, errors='coerce')
        
        # Replace inf/-inf with NaN, then fill NaN with column mean
        X_train0 = X_train0.replace([float('inf'), float('-inf')], float('nan'))
        X_train0 = X_train0.fillna(X_train0.mean())

        X_train0_scaled = scaler.transform(X_train0)
        X_train0_scaled = X_train0_scaled.reshape((X_train0_scaled.shape[0], X_train0_scaled.shape[1], 1))

        # Local model
        local_model = build_model_tuner(hp, input_dim=features)
        #local_model = Simple_local.build(features, classes)
        local_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        local_model.set_weights(global_weights)

        local_model.fit(X_train0_scaled, ytr, epochs=epochs, verbose=1, batch_size=32)

        scaled_weights = local_model.get_weights()
        for i in range(len(scaled_weights)):
            scaled_local_weight_list[i] += scaled_weights[i]

    # Averaging
    average_weights = [w / num_customers for w in scaled_local_weight_list]
    global_model.set_weights(average_weights)

    # Evaluate on central test data
    _, accuracy = global_model.evaluate(x_test, y_test, verbose=0)
    print(f"Round {comm_round + 1} - Global Accuracy: {accuracy:.4f}")
