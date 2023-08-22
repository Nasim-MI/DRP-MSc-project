import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers

# Tensorflow neural network models using one-hot encoded SMILES or molecular fingerprints as drug representation

## Phosphoproteomics model using one-hot encoded SMILES
def build_Phos_SMILES_CNN(learning_rate=1e-3, momentum=0.9, seed=42):

    # set weight initialiser
    initializer = tf.keras.initializers.GlorotUniform(seed)
    
    # omics data input
    x_input = layers.Input(shape=(xo_train.shape[1],1))
    x = layers.Dropout(0.1)(x_input)
    # 1st convolution layer
    x = layers.Conv1D(filters=8, kernel_size=4, kernel_initializer=initializer, activation='relu')(x) 
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D()(x)
    x = layers.Dropout(0.1)(x)
    # 2nd convolution layer
    x = layers.Conv1D(filters=16, kernel_size=4, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D()(x)
    x = layers.Dropout(0.1)(x)
    # 3rd convolution layer
    x = layers.Conv1D(filters=16, kernel_size=4, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Flatten()(x)
    
    
    # one-hot encoded drug data input (default 1 layer)
    y_input = layers.Input(shape=(xd_vals.shape[1:]))
    y = layers.Dropout(0.1)(y_input)
    # 1st convolution layer
    y = layers.Conv1D(filters=75, kernel_size=11, kernel_initializer=initializer, activation='relu')(y) 
    y = layers.BatchNormalization()(y)
    y = layers.MaxPooling1D(pool_size=3, strides=3)(y)
    y = layers.Dropout(0.1)(y)
    y = layers.Flatten()(y) 
    
    # FC layer for xd_train
    y = layers.Dense(64, kernel_initializer=initializer, activation='relu')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dense(64, kernel_initializer=initializer, activation='relu')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.5)(y)
    
    
    # Concatenate omics and encoded drug data
    z = layers.concatenate([x, y])
    z = layers.Dense(64, kernel_initializer=initializer, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dense(32, kernel_initializer=initializer, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dense(1, kernel_initializer=initializer)(z)

    model = tf.keras.Model([x_input, y_input], z)
    
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate, 
                                                        momentum=momentum), 
                                                        loss='mse', metrics=['mae'])
    return model



## Phosphoproteomics model using molecular fingerprints
def build_model(learning_rate=1e-3, momentum=0.9, seed=42):

    # set weight initialiser
    initializer = tf.keras.initializers.GlorotUniform(seed)
    
    # omics data input
    x_input = layers.Input(shape=(xo_train.shape[1],1))
    x = layers.Dropout(0.1)(x_input)
    # 1st convolution layer
    x = layers.Conv1D(filters=8, kernel_size=4, kernel_initializer=initializer, activation='relu')(x) 
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D()(x)
    x = layers.Dropout(0.1)(x)
    # 2nd convolution layer
    x = layers.Conv1D(filters=16, kernel_size=4, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D()(x)
    x = layers.Dropout(0.1)(x)
    # 3rd convolution layer
    x = layers.Conv1D(filters=16, kernel_size=4, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Flatten()(x)
    
    
    # one-hot encoded drug data input (default 3 layers)
    y_input = layers.Input(shape=(xd_test.shape[1],1))

    # 1st convolution layer
    y = layers.Conv1D(filters=64, kernel_size=11, kernel_initializer=initializer, activation='relu')(y_input) 
    y = layers.BatchNormalization()(y)
    y = layers.MaxPooling1D()(y)
    y = layers.Dropout(0.1)(y)
    y = layers.Conv1D(filters=64, kernel_size=11, kernel_initializer=initializer, activation='relu')(y) 
    y = layers.BatchNormalization()(y)
    y = layers.MaxPooling1D()(y)
    y = layers.Dropout(0.1)(y)
    y = layers.Conv1D(filters=64, kernel_size=11, kernel_initializer=initializer, activation='relu')(y) 
    y = layers.BatchNormalization()(y)
    y = layers.MaxPooling1D()(y)
    y = layers.Dropout(0.1)(y)
    y = layers.Flatten()(y) 
    
    # FC layer for xd_train
    y = layers.Dense(64, kernel_initializer=initializer, activation='relu')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dense(64, kernel_initializer=initializer, activation='relu')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.5)(y)
    
    
    # Concatenate omics and encoded drug data
    z = layers.concatenate([x, y])
    z = layers.Dense(64, kernel_initializer=initializer, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dense(64, kernel_initializer=initializer, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dense(1, kernel_initializer=initializer)(z)

    model = tf.keras.Model([x_input, y_input], z)
    
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate, 
                                                        momentum=momentum), 
                                                        loss='mse', metrics=['mae'])
    return model