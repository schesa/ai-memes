import os
import json
import string
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from functools import reduce
from random import random
from PIL import Image
from time import time
import matplotlib.pyplot as plt

import tensorflow.keras.applications.inception_v3
import tensorflow.keras.preprocessing.image
from tensorflow.keras.applications.inception_v3 import InceptionV3
import tensorflow.keras.models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import Input, layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import random

class Model():
    WIDTH = None
    HEIGHT = None
    OUTPUT_DIM = None
    encode_model = None
    preprocess_input= None
    caption_model = None

    loader = None

    def __init__(self):
        self.WIDTH = 299
        self.HEIGHT = 299
        self.OUTPUT_DIM = 2048
        self.load_CNN()

    def load_CNN(self):
        self.encode_model = InceptionV3(weights='imagenet')
        self.encode_model = tensorflow.keras.models.Model(self.encode_model.input, self.encode_model.layers[-2].output)
        self.preprocess_input = tensorflow.keras.applications.inception_v3.preprocess_input
        print('InceptionV3 loaded!')

    def encodeImage(self, img):
        # Resize all images to a standard size (specified bythe image encoding network)
        img = img.resize((self.WIDTH, self.HEIGHT), Image.ANTIALIAS)
        # Convert a PIL image to a numpy array
        x = tensorflow.keras.preprocessing.image.img_to_array(img)
        # Expand to 2D array
        x = np.expand_dims(x, axis=0)
        # Perform any preprocessing needed by InceptionV3 or others
        x = self.preprocess_input(x)
        # Call InceptionV3 (or other) to extract the smaller feature set for the image.
        x = self.encode_model.predict(x) # Get the encoding vector for the image
        # Shape to correct form to be accepted by LSTM captioning network.
        x = np.reshape(x, self.OUTPUT_DIM )
        return x

    def build(self, loader):
        self.loader = loader
        # build Network
        print('Building model')
        inputs1 = Input(shape=(self.OUTPUT_DIM,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)
        inputs2 = Input(shape=(loader.max_length,))
        se1 = Embedding(loader.vocab_size, loader.embedding_dim, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)
        decoder1 = add([fe2,se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(loader.vocab_size, activation='softmax')(decoder2)
        self.caption_model = tensorflow.keras.models.Model(inputs=[inputs1,inputs2], outputs=outputs)
        self.caption_model.layers[2].set_weights([loader.embedding_matrix])
        self.caption_model.layers[2].trainable = False
        print('Compiling model')
        self.caption_model.compile(loss='categorical_crossentropy', optimizer='adam')


    def random_data_generator(self, photos, wordtoidx, max_length, num_descriptions_per_batch, train_data):
        # x1 - Training data for photos
        # x2 - The caption that goes with each photo
        # y - The predicted rest of the caption
        x1, x2, y = [], [], []
        while True:
            random.shuffle(train_data)
            d=0
            for key, desc in train_data:
                photo = photos[key+'.jpg']
                d+=1
                if d==num_descriptions_per_batch:
                    yield ([np.array(x1), np.array(x2)], np.array(y))
                    x1, x2, y = [], [], []
                    d=0
                # Convert each word into a list of sequences.
                seq = [wordtoidx[word] for word in desc.split(' ') if word in wordtoidx]
                # Generate a training case for every possible sequence and outcome
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=self.loader.vocab_size)[0]
                    x1.append(photo)
                    x2.append(in_seq)
                    y.append(out_seq)
            
    def train(self):
        # load weights
        train_data = [(key, desc) for key, desc_list in self.loader.train_descriptions.items() for desc in desc_list ]
        model_path = os.path.join(self.loader.root_path,'ImgFlip500K_Dataset',"data",'caption-model.hdf5')
        if not os.path.exists(model_path):
            print('Training..')
            EPOCHS = 1
            start = time()
            num_descriptions_per_batch=32
            steps = (len(self.loader.all_train_captions))//num_descriptions_per_batch-1
            print(f'steps: {steps}')
            
            generator = self.random_data_generator(self.loader.encoding_train, self.loader.wordtoidx, self.loader.max_length, num_descriptions_per_batch, train_data)
            print(f'\n Fitting Steps1 {steps}\n')
            self.caption_model.fit(generator, epochs=EPOCHS, steps_per_epoch=steps, verbose=1) #, callbacks=[self.get_callback(2)])

            self.caption_model.save_weights(model_path)
            print(f"\Training took: {self.loader.hms_string(time()-start)}")
        else:
            self.caption_model.load_weights(model_path)
        print('Model loaded')
