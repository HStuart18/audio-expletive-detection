import numpy as np
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import load_model, Sequential
from keras.models import Model as KModel
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from ast import literal_eval as make_tuple
import math
import random
import sys
import os

print(K.tensorflow_backend._get_available_gpus())

### HYPERPARAMETERS
# Training params
N_SEARCHES = 1
EPOCHS = [1, 1, 1, 1, 1]
BATCH_SIZE = [8, 5, 5, 5, 5]
ALPHA = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
BETA_1 = [0.9, 0.9, 0.9, 0.9, 0.9]
BETA_2 = [0.999, 0.999, 0.999, 0.999, 0.999]
LAMBDA = [0.01, 0.01, 0.01, 0.01, 0.01]

# Model architecture
KERNEL_SIZE = [8, 8, 8, 8, 8]
STRIDE_LENGTH = [1, 1, 1, 1, 1]
N_FILTERS = [196, 196, 196, 196, 196]
N_UNITS = [128, 128, 128, 128, 128]

# Data
Tx = 799
N_FREQS = 161
CLIP_DUR = 10000
Ty = []
y_INTERVAL = []
for i in range(N_SEARCHES):
    Ty.append(int(((Tx - KERNEL_SIZE[i]) / STRIDE_LENGTH[i]) + 1))
    y_INTERVAL.append(CLIP_DUR / Ty[-1])
N_DEVS = 2
N_VALS = 2

### DEFINE MODEL ARCHITECTURE AND CONFIGURE SETTINGS
def Model(input_shape, f, k, s, u):
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """
    
    X_input = Input(shape = input_shape)
    
    ### START CODE HERE ###
    
    # Step 1: CONV layer (≈4 lines)
    X = Conv1D(f, kernel_size=k, strides=s)(X_input)                                 # CONV1D
    X = BatchNormalization()(X)                                 # Batch normalization
    X = Activation('relu')(X)                                 # ReLu activation
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)

    # Step 2: First GRU Layer (≈4 lines)
    X = GRU(units = u, return_sequences = True)(X) # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)
    X = BatchNormalization()(X)                                 # Batch normalization
    
    # Step 3: Second GRU Layer (≈4 lines)
    X = GRU(units = u, return_sequences = True)(X)   # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)
    X = BatchNormalization()(X)                                  # Batch normalization
    X = Dropout(0.8)(X)                                  # dropout (use 0.8)
    
    # Step 4: Time-distributed dense layer (≈1 line)
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X) # time distributed  (sigmoid)

    ### END CODE HERE ###

    model = KModel(inputs = X_input, outputs = X)
    
    return model

def generate_set(indices, spec_dir, ts_dir, k, y_interval, Ty):
    X = []
    Y = []
    for i in indices:
        x = np.loadtxt(spec_dir + f"\{i}.txt")
        X.append(np.transpose(x))
        final_timestamps = []
        with open(ts_dir + f"\{i}.txt") as f:
            v = f.read().splitlines()
            for t in v:
                final_timestamps.append(make_tuple(t)[1])
        y = [0] * Ty
        # Set index corresponding to the end of timestamp for positive 
        # sample and subsequent k indices to 1
        for t in final_timestamps:
            ind = math.ceil(t / y_interval)
            if ind > Ty:
                ind = Ty
            j = k if Ty - ind >= k else Ty - ind
            for e in range(j + 1):
                y[ind - 1 + e] = 1
        y = np.array([y]).T
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    return (X, Y)

def batch_generator(indices, n, Ty, y_interval, ts_dir, spec_dir, k=15, pos_w=0.8):
    """
    Generates a batch for model training. (input, target, weights)
    """
    while True:
        X=[]
        Y=[]
        W = []

        for i in random.sample(indices, n):
            final_timestamps = []
            with open(ts_dir + f"\{i}.txt") as f:
                v = f.read().splitlines()
                for t in v:
                    final_timestamps.append(make_tuple(t)[1])
            y = [0] * Ty
            w = [1 - pos_w] * Ty
            for t in final_timestamps:
                ind = math.ceil(t / y_interval)
                if ind > Ty:
                    ind = Ty
                j = k if Ty - ind >= k else Ty - ind
                for e in range(j + 1):
                    y[ind - 1 + e] = 1
                    w[ind - 1 + e] = pos_w
            y = np.array([y]).T
            Y.append(y)
            w = np.array(w).T
            W.append(w)
            x = np.loadtxt(spec_dir + f"\{i}.txt")
            X.append(np.transpose(x))

        X = np.array(X)
        Y = np.array(Y)
        W = np.array(W)
        yield (X, Y, W)

def recall(y_true, y_pred):
    """
    What proportion of actual positives was identified correctly?
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    """
    What proportion of positive identifications was actually correct?
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def fbeta_score(y_true, y_pred, beta=0.9):
    # weighted mean of the proportion of correct class
    # assignments vs. the proportion of incorrect class assignments
    # 1 is good. 0 is bad.

    ## If there are no true positives, penalise high misclassifications.
    #if K.sum(y_true) == 0:
        #return 1 - (K.sum(y_pred) / Ty)

    # If there are no true positives, return 0
    if K.sum(y_true) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

class EarlyStoppingByFbeta(Callback):
    """
    Stop early if val_fbeta_score is below threshold value after
    certain number of epochs.
    """
    def __init__(self, monitor="val_fbeta_score", value=0.6):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value

    def on_epoch_end(self, epoch, logs={}, ):
        current = logs.get(self.monitor)
        if current < self.value and epoch >= 50:
            print(f"Stopping early since {current} < {self.value}")
            self.model.stop_training = True

# Only audio with 2 or more positive examples
available_indices = []
for t in os.listdir(r"C:\Users\Harry\Desktop\timestamps"):
    with open(r"C:\Users\Harry\Desktop\timestamps" + f"\{t}") as f:
        v = f.read().splitlines()
        if len(v) > 1:
            available_indices.append(int(t[:-4]))
N_SAMPLES = len(available_indices)
dev_indices = available_indices[:N_DEVS]
val_indices = available_indices[N_DEVS:N_DEVS + N_VALS]
batch_indices = available_indices[N_DEVS + N_VALS:]

for j in range(N_SEARCHES):
    ### GENERATE DEVELOPMENT AND VALIDATION SETS
    X_dev, Y_dev = generate_set(dev_indices, r"C:\Users\Harry\Desktop\spectrograms", 
                                r"C:\Users\Harry\Desktop\timestamps", 15, y_INTERVAL[j], Ty[j])
    X_val, Y_val = generate_set(val_indices, r"C:\Users\Harry\Desktop\spectrograms", 
                                r"C:\Users\Harry\Desktop\timestamps", 15, y_INTERVAL[j], Ty[j])

    #dependencies = {
    #'fbeta_score': fbeta_score
    #}

    #model = load_model(r"C:\Users\Harry\source\repos\HStuart18\audio-expletive-detection\results\models\0\model1-5.hdf5", custom_objects=dependencies)

    #results = model.predict(X_dev)

    #for i in range(N_DEVS):
        #r = list(results[i].T[0])
        #r = np.array([0 if e < 0.5 else 1 for e in r])
        #if not os.path.exists(rf"results/test/{j}"):
            #os.makedirs(rf"results/test/{j}")
        #with open(rf"results/test/{j}/{i}.txt", "w+") as f:
            #f.write(np.array2string(Y_dev[i].T))
            #f.write("\n")
            #f.write(np.array2string(r))

    print("Dev and val sets created")

    ### INITIALISE MODEL
    # Initialise model with callbacks
    opt = Adam(lr=ALPHA[j], beta_1=BETA_1[j], beta_2=BETA_2[j], decay=LAMBDA[j])

    model = Model(input_shape = (Tx, N_FREQS), f=N_FILTERS[j], k=KERNEL_SIZE[j], s=STRIDE_LENGTH[j], u=N_UNITS[j])

    g = batch_generator(batch_indices, BATCH_SIZE[j], Ty[j], y_INTERVAL[j], r"C:\Users\Harry\Desktop\timestamps",
                        r"C:\Users\Harry\Desktop\spectrograms")

    print(K.tensorflow_backend._get_available_gpus())
    print(model.summary())
    print("Beginning model training")

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy", fbeta_score], sample_weight_mode="temporal")

    if not os.path.exists("results\models" + f"\{j}"):
        os.makedirs("results\models" + f"\{j}")

    checkpoint = ModelCheckpoint("results\models" + f"\{j}" + "\model1-{epoch}.hdf5",
                                 monitor='val_loss',
                                 mode='min',
                                 save_best_only=True,
                                 verbose=1)

    esfbeta = EarlyStoppingByFbeta()
    esloss = EarlyStopping(monitor='val_loss', patience=2, mode='min')
    #int(N_SAMPLES/BATCH_SIZE[j])
    model.fit(g, steps_per_epoch=150, epochs=EPOCHS[j], callbacks=[checkpoint, esfbeta, esloss], 
              validation_freq=1, validation_data=(X_val, Y_val))

    loss, acc, fbeta = model.evaluate(X_dev, Y_dev)
    print("Dev set loss = ", loss)
    print("Dev set accuracy = ", acc)
    print("Dev set fbeta_score = ", fbeta)

    results = model.predict(X_dev)

    for i in range(N_DEVS):
        r = list(results[i].T[0])
        r = np.array([0 if e < 0.5 else 1 for e in r])
        if not os.path.exists(rf"results/test/{j}"):
            os.makedirs(rf"results/test/{j}")
        with open(rf"results/test/{j}/{i}.txt", "w+") as f:
            f.write(np.array2string(Y_dev[i].T))
            f.write("\n")
            f.write(np.array2string(r))

    print(f"Search number {j} complete\n")
    


