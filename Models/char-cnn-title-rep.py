import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Input, Embedding, Activation, Flatten, Dense
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.models import load_model


train_data_source = '2.csv'
test_data_source = '2test.csv'


train_df = pd.read_csv(train_data_source, header=None)
test_df = pd.read_csv(test_data_source, header=None)


# concatenate column 1 and column 2 as one text
for df in [train_df,test_df]:
    df[4] = df[4] + df[5]
    df = df.drop([5], axis=1)

# convert string to lower case
train_texts = train_df[4].values
train_texts = [s.lower() for s in train_texts]

test_texts = test_df[4].values
test_texts = [s.lower() for s in test_texts]


# Tokenizer
tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
tk.fit_on_texts(train_texts)


# construct a new vocabulary
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 ,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
char_dict = {}
for i, char in enumerate(alphabet):
    char_dict[char] = i + 1

# Use char_dict to replace the tk.word_index
tk.word_index = char_dict.copy()
# Add 'UNK' to the vocabulary
tk.word_index[tk.oov_token] = max(char_dict.values()) + 1

# Convert string to index
train_sequences = tk.texts_to_sequences(train_texts)
test_texts = tk.texts_to_sequences(test_texts)

# Padding
train_data = pad_sequences(train_sequences, maxlen=1014, padding='post')
test_data = pad_sequences(test_texts, maxlen=1014, padding='post')

# Convert to numpy array
train_data = np.array(train_data, dtype='float32')
test_data = np.array(test_data, dtype='float32')


train_classes = train_df[2].values
train_class_list = [int(x)-1 for x in train_classes]

test_classes = test_df[2].values
test_class_list = [int(x)-1 for x in test_classes]


train_classes = to_categorical(train_class_list)
test_classes = to_categorical(test_class_list)


vocab_size = len(tk.word_index)
embedding_weights = [] #(71, 70)
embedding_weights.append(np.zeros(vocab_size)) # first row is pad

for char, i in tk.word_index.items(): # from index 1 to 70
    onehot = np.zeros(vocab_size)
    onehot[i-1] = 1
    embedding_weights.append(onehot)
embedding_weights = np.array(embedding_weights)
print(embedding_weights.shape)

# parameter
input_size = 1014
# vocab_size = 69
embedding_size = 70
conv_layers = [[256, 7, 3],
               [256, 7, 3],
               [256, 3, -1],
               [256, 3, -1],
               [256, 3, -1],
               [256, 3, 3]]

fully_connected_layers = [1024, 1024]
num_of_classes = 3
dropout_p = 0.5
optimizer = 'adam'
loss = 'categorical_crossentropy'

# Embedding layer Initialization
embedding_layer = Embedding(vocab_size+1,
                            embedding_size,
                            input_length=input_size,
                            weights=[embedding_weights])

# parameter
input_size = 1014
vocab_size = len(tk.word_index)
embedding_size = 70
conv_layers = [[256, 7, 3],
               [256, 7, 3],
               [256, 3, -1],
               [256, 3, -1],
               [256, 3, -1],
               [256, 3, 3]]

fully_connected_layers = [1024, 1024]
num_of_classes = 3
dropout_p = 0.5
optimizer = 'adam'
loss = 'categorical_crossentropy'

# Embedding weights
embedding_weights = []  # (71, 70)
embedding_weights.append(np.zeros(vocab_size))  # (0, 70)

for char, i in tk.word_index.items():  # from index 1 to 70
    onehot = np.zeros(vocab_size)
    onehot[i - 1] = 1
    embedding_weights.append(onehot)

embedding_weights = np.array(embedding_weights)
print('Load')

# Embedding layer Initialization
embedding_layer = Embedding(vocab_size + 1,
                            embedding_size,
                            input_length=input_size,
                            weights=[embedding_weights])

# Model Construction
# Input
inputs = Input(shape=(input_size,), name='input', dtype='int64')  # shape=(?, 1014)
# Embedding
x = embedding_layer(inputs)
# Conv
for filter_num, filter_size, pooling_size in conv_layers:
    x = Conv1D(filter_num, filter_size,kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
    x = Activation('relu')(x)
    if pooling_size != -1:
        x = MaxPooling1D(pool_size=pooling_size)(x)  # Final shape=(None, 34, 256)
x = Flatten()(x)  # (None, 8704)
# Fully connected layers
for dense_size in fully_connected_layers:
    x = Dense(dense_size, activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)  # dense_size == 1024
    x = Dropout(dropout_p)(x)
# Output Layer
predictions = Dense(num_of_classes, activation='softmax')(x)
# Build model
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])  # Adam, categorical_crossentropy
model.summary()
# Shuffle
indices = np.arange(train_data.shape[0])
np.random.shuffle(indices)

x_train = train_data[indices]
y_train = train_classes[indices]

x_test = test_data
y_test = test_classes


# Training
model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          batch_size=128,
          epochs=10,
          verbose=2)

model.save('title-representativeness.h5')