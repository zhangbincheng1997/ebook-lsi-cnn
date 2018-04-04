from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, Flatten, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, GlobalAveragePooling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed

####################
MAX_SEQUENCE_LENGTH = 200  # 问题/答案 上限200个词
EMBEDDING_DIM = 100  # 100d 词向量

QA_EMBED_SIZE = 64
DROPOUT_RATE = 0.5
####################


import pickle

token_path = 'data/tokenizer.pkl'
tokenizer = pickle.load(open(token_path, 'rb'))
word_index = tokenizer.word_index


def embedding_raw():
    return Embedding(len(word_index) + 1,
                     EMBEDDING_DIM,
                     input_length=MAX_SEQUENCE_LENGTH)


def embedding():
    import numpy as np
    embedding_matrix = np.load('data/embedding_matrix.npy')
    return Embedding(len(word_index) + 1,
                     EMBEDDING_DIM,
                     weights=[embedding_matrix],
                     input_length=MAX_SEQUENCE_LENGTH,
                     trainable=False)


def FastText():
    input_q = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float64')
    q = embedding_raw()(input_q)
    q = GlobalAveragePooling1D()(q)
    q = Dropout(DROPOUT_RATE)(q)

    input_a = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float64')
    a = embedding_raw()(input_a)
    a = GlobalAveragePooling1D()(a)
    a = Dropout(DROPOUT_RATE)(a)

    merged = concatenate([q, a])
    merged = Dense(64)(merged)
    merged = BatchNormalization()(merged)
    merged = Activation('relu')(merged)
    merged = Dropout(DROPOUT_RATE)(merged)
    merged = Dense(1, activation="sigmoid")(merged)

    model = Model([input_q, input_a], [merged])
    return model


def CNN1():
    ##### Q
    input_q = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float64')
    q = embedding_raw()(input_q)
    # cnn1模块，kernel_size = 3
    conv1_1 = Conv1D(256, 3, padding='same')(q)
    bn1_1 = BatchNormalization()(conv1_1)
    relu1_1 = Activation('relu')(bn1_1)
    conv1_2 = Conv1D(128, 3, padding='same')(relu1_1)
    bn1_2 = BatchNormalization()(conv1_2)
    relu1_2 = Activation('relu')(bn1_2)
    cnn1 = MaxPooling1D(pool_size=2)(relu1_2)
    # cnn2模块，kernel_size = 4
    conv2_1 = Conv1D(256, 4, padding='same')(q)
    bn2_1 = BatchNormalization()(conv2_1)
    relu2_1 = Activation('relu')(bn2_1)
    conv2_2 = Conv1D(128, 4, padding='same')(relu2_1)
    bn2_2 = BatchNormalization()(conv2_2)
    relu2_2 = Activation('relu')(bn2_2)
    cnn2 = MaxPooling1D(pool_size=2)(relu2_2)
    # cnn3模块，kernel_size = 5
    conv3_1 = Conv1D(256, 5, padding='same')(q)
    bn3_1 = BatchNormalization()(conv3_1)
    relu3_1 = Activation('relu')(bn3_1)
    conv3_2 = Conv1D(128, 5, padding='same')(relu3_1)
    bn3_2 = BatchNormalization()(conv3_2)
    relu3_2 = Activation('relu')(bn3_2)
    cnn3 = MaxPooling1D(pool_size=2)(relu3_2)
    # 拼接三个模块
    cnn_q = concatenate([cnn1, cnn2, cnn3], axis=-1)
    cnn_q = Flatten()(cnn_q)
    cnn_q = Dropout(DROPOUT_RATE)(cnn_q)

    ##### A
    input_a = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float64')
    a = embedding_raw()(input_a)
    # cnn1模块，kernel_size = 3
    conv1_1 = Conv1D(256, 3, padding='same')(a)
    bn1_1 = BatchNormalization()(conv1_1)
    relu1_1 = Activation('relu')(bn1_1)
    conv1_2 = Conv1D(128, 3, padding='same')(relu1_1)
    bn1_2 = BatchNormalization()(conv1_2)
    relu1_2 = Activation('relu')(bn1_2)
    cnn1 = MaxPooling1D(pool_size=2)(relu1_2)
    # cnn2模块，kernel_size = 4
    conv2_1 = Conv1D(256, 4, padding='same')(a)
    bn2_1 = BatchNormalization()(conv2_1)
    relu2_1 = Activation('relu')(bn2_1)
    conv2_2 = Conv1D(128, 4, padding='same')(relu2_1)
    bn2_2 = BatchNormalization()(conv2_2)
    relu2_2 = Activation('relu')(bn2_2)
    cnn2 = MaxPooling1D(pool_size=2)(relu2_2)
    # cnn3模块，kernel_size = 5
    conv3_1 = Conv1D(256, 5, padding='same')(a)
    bn3_1 = BatchNormalization()(conv3_1)
    relu3_1 = Activation('relu')(bn3_1)
    conv3_2 = Conv1D(128, 5, padding='same')(relu3_1)
    bn3_2 = BatchNormalization()(conv3_2)
    relu3_2 = Activation('relu')(bn3_2)
    cnn3 = MaxPooling1D(pool_size=2)(relu3_2)
    # 拼接三个模块
    cnn_a = concatenate([cnn1, cnn2, cnn3], axis=-1)
    cnn_a = Flatten()(cnn_a)
    cnn_a = Dropout(DROPOUT_RATE)(cnn_a)

    ###### Q-A
    merged = concatenate([cnn_q, cnn_a])
    merged = Dense(512)(merged)
    merged = BatchNormalization()(merged)
    merged = Activation('relu')(merged)
    merged = Dropout(DROPOUT_RATE)(merged)
    merged = Dense(1, activation="sigmoid")(merged)

    model = Model([input_q, input_a], [merged])
    return model


def CNN2():
    ##### Q
    input_q = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float64')
    q = embedding()(input_q)
    # cnn1模块，kernel_size = 3
    conv1_1 = Conv1D(256, 3, padding='same')(q)
    bn1_1 = BatchNormalization()(conv1_1)
    relu1_1 = Activation('relu')(bn1_1)
    conv1_2 = Conv1D(128, 3, padding='same')(relu1_1)
    bn1_2 = BatchNormalization()(conv1_2)
    relu1_2 = Activation('relu')(bn1_2)
    cnn1 = MaxPooling1D(pool_size=2)(relu1_2)
    # cnn2模块，kernel_size = 4
    conv2_1 = Conv1D(256, 4, padding='same')(q)
    bn2_1 = BatchNormalization()(conv2_1)
    relu2_1 = Activation('relu')(bn2_1)
    conv2_2 = Conv1D(128, 4, padding='same')(relu2_1)
    bn2_2 = BatchNormalization()(conv2_2)
    relu2_2 = Activation('relu')(bn2_2)
    cnn2 = MaxPooling1D(pool_size=2)(relu2_2)
    # cnn3模块，kernel_size = 5
    conv3_1 = Conv1D(256, 5, padding='same')(q)
    bn3_1 = BatchNormalization()(conv3_1)
    relu3_1 = Activation('relu')(bn3_1)
    conv3_2 = Conv1D(128, 5, padding='same')(relu3_1)
    bn3_2 = BatchNormalization()(conv3_2)
    relu3_2 = Activation('relu')(bn3_2)
    cnn3 = MaxPooling1D(pool_size=2)(relu3_2)
    # 拼接三个模块
    cnn_q = concatenate([cnn1, cnn2, cnn3], axis=-1)
    cnn_q = Flatten()(cnn_q)
    cnn_q = Dropout(DROPOUT_RATE)(cnn_q)

    ##### A
    input_a = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float64')
    a = embedding()(input_a)
    # cnn1模块，kernel_size = 3
    conv1_1 = Conv1D(256, 3, padding='same')(a)
    bn1_1 = BatchNormalization()(conv1_1)
    relu1_1 = Activation('relu')(bn1_1)
    conv1_2 = Conv1D(128, 3, padding='same')(relu1_1)
    bn1_2 = BatchNormalization()(conv1_2)
    relu1_2 = Activation('relu')(bn1_2)
    cnn1 = MaxPooling1D(pool_size=2)(relu1_2)
    # cnn2模块，kernel_size = 4
    conv2_1 = Conv1D(256, 4, padding='same')(a)
    bn2_1 = BatchNormalization()(conv2_1)
    relu2_1 = Activation('relu')(bn2_1)
    conv2_2 = Conv1D(128, 4, padding='same')(relu2_1)
    bn2_2 = BatchNormalization()(conv2_2)
    relu2_2 = Activation('relu')(bn2_2)
    cnn2 = MaxPooling1D(pool_size=2)(relu2_2)
    # cnn3模块，kernel_size = 5
    conv3_1 = Conv1D(256, 5, padding='same')(a)
    bn3_1 = BatchNormalization()(conv3_1)
    relu3_1 = Activation('relu')(bn3_1)
    conv3_2 = Conv1D(128, 5, padding='same')(relu3_1)
    bn3_2 = BatchNormalization()(conv3_2)
    relu3_2 = Activation('relu')(bn3_2)
    cnn3 = MaxPooling1D(pool_size=2)(relu3_2)
    # 拼接三个模块
    cnn_a = concatenate([cnn1, cnn2, cnn3], axis=-1)
    cnn_a = Flatten()(cnn_a)
    cnn_a = Dropout(DROPOUT_RATE)(cnn_a)

    ###### Q-A
    merged = concatenate([cnn_q, cnn_a])
    merged = Dense(512)(merged)
    merged = BatchNormalization()(merged)
    merged = Activation('relu')(merged)
    merged = Dropout(DROPOUT_RATE)(merged)
    merged = Dense(1, activation="sigmoid")(merged)

    model = Model([input_q, input_a], [merged])
    return model


def BiLSTM():
    input_q = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float64')
    q = embedding()(input_q)
    q = Bidirectional(LSTM(QA_EMBED_SIZE, return_sequences=True, dropout=DROPOUT_RATE, recurrent_dropout=DROPOUT_RATE),
                      merge_mode="sum")(q)
    q = TimeDistributed(Dense(QA_EMBED_SIZE))(q)
    q = Flatten()(q)
    q = Dropout(DROPOUT_RATE)(q)

    input_a = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float64')
    a = embedding()(input_a)
    a = Bidirectional(LSTM(QA_EMBED_SIZE, return_sequences=True, dropout=DROPOUT_RATE, recurrent_dropout=DROPOUT_RATE),
                      merge_mode="sum")(a)
    a = TimeDistributed(Dense(QA_EMBED_SIZE))(a)
    a = Flatten()(a)
    a = Dropout(DROPOUT_RATE)(a)

    merged = concatenate([q, a])
    merged = Dense(512)(merged)
    merged = BatchNormalization()(merged)
    merged = Activation('relu')(merged)
    merged = Dropout(DROPOUT_RATE)(merged)
    merged = Dense(1, activation="sigmoid")(merged)

    model = Model([input_q, input_a], [merged])
    return model


def Attention():
    # https://github.com/farizrahman4u/seq2seq or https://github.com/datalogue/keras-attention

    import seq2seq
    from seq2seq.models import AttentionSeq2Seq
    
    input_q = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float64')
    q = embedding()(input_q)
    q = AttentionSeq2Seq(input_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH,
                         output_length=MAX_SEQUENCE_LENGTH, output_dim=QA_EMBED_SIZE,
                         depth=1)(q)
    q = Flatten()(q)
    q = Dropout(DROPOUT_RATE)(q)

    input_a = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float64')
    a = embedding()(input_a)
    a = AttentionSeq2Seq(input_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH,
                         output_length=MAX_SEQUENCE_LENGTH, output_dim=QA_EMBED_SIZE,
                         depth=1)(a)
    a = Flatten()(a)
    a = Dropout(DROPOUT_RATE)(a)

    merged = concatenate([q, a])
    merged = Dense(512)(merged)
    merged = BatchNormalization()(merged)
    merged = Activation('relu')(merged)
    merged = Dropout(DROPOUT_RATE)(merged)
    merged = Dense(1, activation="sigmoid")(merged)

    model = Model([input_q, input_a], [merged])
    return model
