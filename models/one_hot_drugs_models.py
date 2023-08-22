import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers

# Tensorflow neural network models using one-hot encoded drug names as drug representation


## Phosphoproteomics model for landmark targets feature selection
def build_CNN_Phos_LM(xo_train,xd_train,learning_rate=7e-3, momentum=0.8, seed=42): 

    # set weight initialiser
    initializer = tf.keras.initializers.GlorotUniform(seed=seed)
    
    # phosphoproteomics data input
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
    
    # one-hot encoded drug data + dense layer
    y_input = layers.Input(shape=(xd_train.shape[1]))
    y = layers.Dense(256, kernel_initializer=initializer, activation="relu")(y_input) 
    
    # Concatenate phosphoproteomics and encoded drug data
    z = layers.concatenate([x, y])
    z = layers.Dense(64, kernel_initializer=initializer, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.25)(z)
    z = layers.Dense(32, kernel_initializer=initializer, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.25)(z)
    z = layers.Dense(1, kernel_initializer=initializer)(z)

    model = tf.keras.Model([x_input, y_input], z)
    
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate, 
                                                        momentum=momentum), 
                                                        loss='mse', metrics=['mae'])
    return model



## Phosphoproteomics model for functional score feature selection
def build_CNN_Phos_FS(xo_train,xd_train,learning_rate=4e-3, momentum=0.3, seed=42):

    # set weight initialiser
    initializer = tf.keras.initializers.GlorotUniform(seed=seed)
    
    # phosphoproteomics data input
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
    
    # one-hot encoded drug data + dense layer
    y_input = layers.Input(shape=(xd_train.shape[1]))
    y = layers.Dense(256, kernel_initializer=initializer, activation="relu")(y_input) 
    
    # Concatenate phosphoproteomics and encoded drug data
    z = layers.concatenate([x, y])
    z = layers.Dense(64, kernel_initializer=initializer, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.25)(z)
    z = layers.Dense(32, kernel_initializer=initializer, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.25)(z)
    z = layers.Dense(1, kernel_initializer=initializer)(z)

    model = tf.keras.Model([x_input, y_input], z)
    
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate, 
                                                        momentum=momentum), 
                                                        loss='mse', metrics=['mae'])
    return model



## Phosphoproteomics model for functional score feature selection + putative false positive phosphosites removed
def build_CNN_Phos_FS_2(xo_train,xd_train,learning_rate=7e-3, momentum=0.3, seed=42):

    # set weight initialiser
    initializer = tf.keras.initializers.GlorotUniform(seed=seed)
    
    # phosphoproteomics data input
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
    
    # one-hot encoded drug data + dense layer
    y_input = layers.Input(shape=(xd_train.shape[1]))
    y = layers.Dense(256, kernel_initializer=initializer, activation="relu")(y_input) 
    
    # Concatenate phosphoproteomics and encoded drug data
    z = layers.concatenate([x, y])
    z = layers.Dense(64, kernel_initializer=initializer, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.25)(z)
    z = layers.Dense(32, kernel_initializer=initializer, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.25)(z)
    z = layers.Dense(1, kernel_initializer=initializer)(z)

    model = tf.keras.Model([x_input, y_input], z)
    
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate, 
                                                        momentum=momentum), 
                                                        loss='mse', metrics=['mae'])
    return model



## CNN using functional API
def build_Phos_Atlas(xo_train,xd_train,learning_rate=7e-3, momentum=0.8, seed=42): 

    # set weight initialiser
    initializer = tf.keras.initializers.GlorotUniform(seed=seed)
    
    # phosphoproteomics data input
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
    
    # one-hot encoded drug data + dense layer
    y_input = layers.Input(shape=(xd_train.shape[1]))
    y = layers.Dense(256, kernel_initializer=initializer, activation="relu")(y_input) 
    
    # Concatenate phosphoproteomics and encoded drug data
    z = layers.concatenate([x, y])
    z = layers.Dense(64, kernel_initializer=initializer, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.25)(z)
    z = layers.Dense(32, kernel_initializer=initializer, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.25)(z)
    z = layers.Dense(1, kernel_initializer=initializer)(z)

    model = tf.keras.Model([x_input, y_input], z)
    
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate, 
                                                        momentum=momentum), 
                                                        loss='mse', metrics=['mae'])
    return model



## Phosphoproteomics model for landmark kinase substrate specificity feature selection method
def build_CNN_Phos_AtlasLM(xo_train,xd_train,learning_rate=7e-3, momentum=0.3, seed=42):

    # set weight initialiser
    initializer = tf.keras.initializers.GlorotUniform(seed=seed)
    
    # phosphoproteomics data input
    x_input = layers.Input(shape=(xo_train.shape[1],1))
    x = layers.Dropout(0.3)(x_input)
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
    
    # one-hot encoded drug data + dense layer
    y_input = layers.Input(shape=(xd_train.shape[1]))
    y = layers.Dense(256, kernel_initializer=initializer, activation="relu")(y_input) 
    
    # Concatenate phosphoproteomics and encoded drug data
    z = layers.concatenate([x, y])
    z = layers.Dense(64, kernel_initializer=initializer, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.25)(z)
    z = layers.Dense(32, kernel_initializer=initializer, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.25)(z)
    z = layers.Dense(1, kernel_initializer=initializer)(z)

    model = tf.keras.Model([x_input, y_input], z)
    
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate, 
                                                        momentum=momentum), 
                                                        loss='mse', metrics=['mae'])
    return model



## Phosphoproteomics model for landmark kinase substrate specificity feature selection method +
def build_CNN_Phos_AtlasLM_2(xo_train,xd_train,learning_rate=7e-3, momentum=0.3, seed=42):

    # set weight initialiser
    initializer = tf.keras.initializers.GlorotUniform(seed=seed)
    
    # phosphoproteomics data input
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
    
    # one-hot encoded drug data + dense layer
    y_input = layers.Input(shape=(xd_train.shape[1]))
    y = layers.Dense(256, kernel_initializer=initializer, activation="relu")(y_input) 
    
    # Concatenate phosphoproteomics and encoded drug data
    z = layers.concatenate([x, y])
    z = layers.Dense(64, kernel_initializer=initializer, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.25)(z)
    z = layers.Dense(32, kernel_initializer=initializer, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.25)(z)
    z = layers.Dense(1, kernel_initializer=initializer)(z)

    model = tf.keras.Model([x_input, y_input], z)
    
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate, 
                                                        momentum=momentum), 
                                                        loss='mse', metrics=['mae'])
    return model



## Proteomics model for landmark proteins
def build_CNN_Prot_LM(xo_train,xd_train,learning_rate=1e-3, momentum=0.6, seed=42): 

    # set weight initialiser
    initializer = tf.keras.initializers.GlorotUniform(seed=seed)
    
    # omics data input
    x_input = layers.Input(shape=(xo_train.shape[1],1))
    x = layers.Dropout(0.1)(x_input)
    # 1st convolution layer
    x = layers.Conv1D(filters=8, kernel_size=1, kernel_initializer=initializer, activation='relu')(x) 
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D()(x)
    x = layers.Dropout(0.1)(x)
    # 2nd convolution layer
    x = layers.Conv1D(filters=16, kernel_size=1, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D()(x)
    x = layers.Dropout(0.1)(x)
    # 3rd convolution layer
    x = layers.Conv1D(filters=32, kernel_size=1, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Flatten()(x)

    
    # one-hot encoded drug data + dense layer
    y_input = layers.Input(shape=(xd_train.shape[1]))
    y = layers.Dense(256, kernel_initializer=initializer, activation="relu")(y_input) 
    
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



## Multi-omics model using both phosphoproteomics and proteomics features
def build_Phos_Prot_CNN(xo_train_phos,xo_train_prot,xd_train,learning_rate=5e-3, momentum=0.7, seed=42):

    # set weight initialiser
    initializer = tf.keras.initializers.GlorotUniform(seed=seed)
    
    ## Drug input
    # one-hot encoded drug data + dense layer
    y_input = layers.Input(shape=(xd_train.shape[1]))
    y = layers.Dense(256, kernel_initializer=initializer, activation="relu")(y_input) 
    
    
    ## Phosphoproteomics branch
    # phosphoproteomics data input
    x_input_phospho = layers.Input(shape=(xo_train_phos.shape[1],1))
    x_1 = layers.Dropout(0.1)(x_input_phospho)
    # 1st convolution layer
    x_1 = layers.Conv1D(filters=8, kernel_size=4, kernel_initializer=initializer, activation='relu')(x_input_phospho) 
    x_1 = layers.BatchNormalization()(x_1)
    x_1 = layers.MaxPooling1D()(x_1)
    x_1 = layers.Dropout(0.1)(x_1)
    # 2nd convolution layer + flatten
    x_1 = layers.Conv1D(filters=16, kernel_size=4, kernel_initializer=initializer, activation='relu')(x_1)
    x_1 = layers.BatchNormalization()(x_1)
    x_1 = layers.MaxPooling1D()(x_1)
    x_1 = layers.Dropout(0.1)(x_1)
    x_1 = layers.Flatten()(x_1)
    
    
    ## Proteomics branch
    # proteomics data input
    x_input_prot = layers.Input(shape=(xo_train_prot.shape[1],1))
    x_2 = layers.Dropout(0.1)(x_input_prot)
    # 1st convolution layer
    x_2 = layers.Conv1D(filters=8, kernel_size=4, kernel_initializer=initializer, activation='relu')(x_input_prot) 
    x_2 = layers.BatchNormalization()(x_2)
    x_2 = layers.MaxPooling1D()(x_2)
    x_2 = layers.Dropout(0.1)(x_2)
    # 2nd convolution layer
    x_2 = layers.Conv1D(filters=16, kernel_size=4, kernel_initializer=initializer, activation='relu')(x_2)
    x_2 = layers.BatchNormalization()(x_2)
    x_2 = layers.MaxPooling1D()(x_2)
    x_2 = layers.Dropout(0.1)(x_2)
    # 3rd convolution layer + flatten
    x_2 = layers.Conv1D(filters=16, kernel_size=4, kernel_initializer=initializer, activation='relu')(x_2)
    x_2 = layers.BatchNormalization()(x_2)
    x_2 = layers.MaxPooling1D()(x_2)
    x_2 = layers.Dropout(0.1)(x_2)
    x_2 = layers.Flatten()(x_2)
    
    
    ## Concatenate omics and encoded drug data
    z = layers.concatenate([x_1, x_2, y])
    z = layers.Dense(64, kernel_initializer=initializer, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.1)(z)
    z = layers.Dense(32, kernel_initializer=initializer, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.1)(z)
    z = layers.Dense(1, kernel_initializer=initializer)(z)

    model = tf.keras.Model([x_input_phospho, x_input_prot, y_input], z)
    
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate, 
                                                        momentum=momentum), 
                                                        loss='mse', metrics=['mae'])
    return model