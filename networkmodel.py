from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers, models
import keras_tuner as kt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout
from tensorflow.keras import Input
from keras import regularizers


def build_model_tuner(hp, input_dim):
    

    model = Sequential()
    model.add(Input(shape=(input_dim, 1)))
    model.add(Conv1D(
        #filters=hp.Int('filters', min_value=16, max_value=128, step=16),
        #kernel_size=hp.Choice('kernel_size', [2, 3, 4]),
        filters=128,
        kernel_size=3,
        kernel_regularizer=regularizers.l2(0.001),
        activation='relu'
    ))
    

    #units1 = hp.Int('units1', min_value=20, max_value=180, step=20)
    #units2 = hp.Int('units2', min_value=20, max_value=180, step=20)
    #units3 = hp.Int('units3', min_value=20, max_value=180, step=20)
    units1 = 150
    units2 = 200
    units3 = 200
    learning_rate = 0.0004571155244
    
    #model.add(Dense(units1, activation='relu', input_dim=input_dim))
    #model.add(Dense(units2, activation='relu'))
    #model.add(Dense(units3, activation='relu'))
    #model.add(Dense(1, activation='sigmoid'))
    model.add(Flatten())
    #units1 = hp.Int('units1', min_value=20, max_value=250, step=20)
    #units2 = hp.Int('units2', min_value=20, max_value=250, step=20)
    #units3 = hp.Int('units3', min_value=20, max_value=250, step=20)
    model.add(Dropout(0.15))
    model.add(Dense(units1, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units2, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units3, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    #  learning rate as a hyperparameter
    #learning_rate = hp.Float('learning_rate', min_value=0.0001, max_value=0.01, sampling='log')
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    #batch_size = hp.Choice('batch_size', [32, 64, 128])
    #epochs = hp.Int('epochs', min_value=50, max_value=3000, step=100)
    batch_size = 64
    epochs = 850

    

    return model