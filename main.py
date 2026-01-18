from preprocess import preprocess_data
from networkmodel import build_model_tuner
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import keras
import keras_tuner as kt
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from tensorflow.keras.callbacks import EarlyStopping
import time  
from tensorflow.keras.layers import Conv1D, Flatten, Dense
import numpy as np
from collections import Counter


drop_columns = ['R2-PM8:V', 'R1-PM8:V', 'R2-PM9:V', 'R4-PA9:VH', 'R2-PA8:VH', 'R3-PA8:VH']  # Columns to drop

x_train, x_test, y_train, y_test, x_val, y_val, scaler = preprocess_data(
    r"/work/projects/reu-2025/c-halkhafaji/cyber security detection model/binaryAllNaturalPlusNormalVsAttacks1",
    'marker',
    drop_columns
)



print("Train class distribution:", Counter(y_train))
print("Validation class distribution:", Counter(y_val))
print("Test class distribution:", Counter(y_test))


# Set input_dim globally for the tuner
import networkmodel
input_dim = x_train.shape[1]


# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


print("x_train shape:", x_train.shape)
print("x_val shape:", x_val.shape)
print("x_test shape:", x_test.shape)

# tuner = kt.RandomSearch(
#     lambda hp: build_model_tuner(hp, input_dim=x_train.shape[1]),
#     objective=kt.Objective("val_accuracy", direction="max"),
#     max_trials=10,
#     executions_per_trial=1,
#     directory='my_dir',
#     project_name='cybersec_tuning'
# )




# # Tune batch size and epochs as hyperparameters
# tuner.search(
#     x_train, y_train,
#     validation_data=(x_val, y_val),
#     epochs = 200,
# )

# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]


# # Build and train the best model
# model = tuner.hypermodel.build(best_hps)
# #model = build_model_tuner(None)
# epochs = best_hps.values.get('epochs')
# batch_size = best_hps.get('batch_size')


# early_stop = EarlyStopping(
#     monitor='val_loss',
#     mode='min',
#     patience=10,
#     min_delta=0.0001,
#     restore_best_weights=True
# )
# start_time = time.time()
# history = model.fit(
#     x_train, y_train,
#     batch_size=best_hps.get('batch_size'),
#     epochs=best_hps.get('epochs'),
#     validation_data=(x_val, y_val),
#     callbacks=[early_stop]
# )

# end_time = time.time()
# print(f"Training time: {end_time - start_time:.2f} seconds")

# # Get the top 5 models from the tuner
# best_models = tuner.get_best_models(num_models=10)
# lowest_far = float('inf')
# best_model = None
# best_model_hpsis = None

# best_model = best_models[0]  # First one is best val_accuracy
# model = best_model

#for i, m in enumerate(best_models):
#     y_pred_probs = m.predict(x_test)
#     y_pred = (y_pred_probs > 0.5).astype(int)
#     tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#     far = fp / (fp + tn) if (fp + tn) > 0 else 0
#     if far < lowest_far:
#         lowest_far = far
#         best_model = m
#         best_model_hps = tuner.oracle.get_trial(tuner.oracle.get_best_trials(num_trials=10)[i].trial_id).hyperparameters.values





#print(f"\nLowest False Alarm Rate from top models: {lowest_far:.4f}")


num_trials = 1
metrics = {
    'loss': [],
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'false_alarm_rate': []
}

for trial in range(num_trials):
    print(f"\n--- Trial {trial + 1} ---")
    model = build_model_tuner(None, input_dim=x_train.shape[1])

    early_stop = EarlyStopping(
         monitor='val_loss',
         mode='min',
         patience=20,
         min_delta=0.001,
         restore_best_weights=True
     )

    history = model.fit(
        x_train, y_train,
        epochs=250,
        batch_size=64,
        validation_data=(x_val, y_val),
        #callbacks=[early_stop],
        verbose=1
    )

    y_pred_probs = model.predict(x_test)
    y_pred = (y_pred_probs > 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    false_alarm_rate = (fp / (fp + tn)) if (fp + tn) > 0 else 0
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    metrics['loss'].append(loss)
    metrics['accuracy'].append(accuracy)
    metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
    metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
    metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
    metrics['false_alarm_rate'].append(false_alarm_rate)

# Print average and standard deviation
print(f"\nAverage over {num_trials} trials:")
for key in metrics:
    avg = np.mean(metrics[key])
    std = np.std(metrics[key])
    print(f"{key}: {avg:.4f} ± {std:.4f}")


# --- Single run reporting and plotting (from the last trial) ---

model.summary()

# # Use the best_model for further evaluation
# model = best_model
# y_pred_probs = model.predict(x_test)
# y_pred = (y_pred_probs > 0.5).astype(int)
# cm = confusion_matrix(y_test, y_pred)
# tn, fp, fn, tp = cm.ravel()
# false_alarm_rate = (fp / (fp + tn)) if (fp + tn) > 0 else 0
# loss, accuracy = model.evaluate(x_test, y_test)

# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# print(f"Test Loss: {loss}")

# # # Only print val_loss if you still have the `history` object from training
# if 'history' in locals():
#     print(f"Final Validation Loss: {history.history['val_loss'][-1]}")
# else:
#     print("Final Validation Loss: (not available — 'history' not in scope)")

# print(f"Test Accuracy: {accuracy}")
# highest_diff = recall_score(y_test, y_pred) - false_alarm_rate
# print("Highest Difference (Recall - FAR):", highest_diff)
# print("False Negatives:", fn)
# print("False Positives:", fp)
# print("Precision:", precision_score(y_test, y_pred))
# print("Recall:", recall_score(y_test, y_pred))
# print("F1 Score:", f1_score(y_test, y_pred))
# print("False Alarm Rate:", false_alarm_rate)

# # # Print best hyperparameters found
# print("\nBest Hyperparameters Found:")
# print(f"Best units: {best_hps.get('units1')}, {best_hps.get('units2')}, {best_hps.get('units3')}")
# print(f"Best batch size: {best_hps.get('batch_size')}")
# print(f"Best epochs: {best_hps.get('epochs')}")
# print(f"Best learning rate: {best_hps.get('learning_rate')}")
# print(f"Best filters: {best_hps.get('filters')}")
# print(f"Best kernel size: {best_hps.get('kernel_size')}")





# Plot only the best model's loss curves
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Test Loss ')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig("training_validation_loss.png")  # Save the figure to a file
plt.show()






# Plot only the best model's loss curves
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Test Loss ')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig("training_validation_loss.png")  # Save the figure to a file
plt.show()
