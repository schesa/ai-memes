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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import Input, layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical 
from .loader import Loader
from .model import Model

class Predictor():
    loader = None
    model = None

    def generateCaption(self, photo, model):
        in_text = self.loader.START
        for _ in range(self.loader.max_length):
            sequence = [self.loader.wordtoidx[w] for w in in_text.split() if w in self.loader.wordtoidx]
            sequence = pad_sequences([sequence], maxlen=self.loader.max_length)
            yhat = model.caption_model.predict([photo,sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = self.loader.idxtoword[yhat]
            in_text += ' ' + word
            print(word)
            if word == self.loader.STOP:
                break
        final = in_text.split()
        final = final[1:-1]
        final = ' '.join(final)
        return final

    def generateCaptions(self):
        # generate captions
        for img in self.loader.train_descriptions.keys():
            image = self.loader.encoding_train[f'{img}.jpg'].reshape((1,self.model.OUTPUT_DIM))
            x = plt.imread(os.path.join(self.loader.root_path,'ImgFlip500K_Dataset','templates','img', f'{img}.jpg'))
            plt.imshow(x)
            plt.show()
            print(image)
            print("Caption:",self.generateCaption(image, self.model))
            print("_____________________________________")

    def get_filename(self, title):
        table = str.maketrans(dict.fromkeys(string.punctuation))
        return "-".join(title.translate(table).split())

    def predict(self, templateid):
        # predict caption
        title = self.loader.get_meme_name(templateid)
        print(f'Title {title}')
        img = self.get_filename(title)
        print(f'Img {img}')
        image = self.loader.encoding_train[f'{img}.jpg'].reshape((1,self.model.OUTPUT_DIM))
        caption = self.generateCaption(image, self.model)
        x = plt.imread(os.path.join(self.loader.root_path,'ImgFlip500K_Dataset','templates','img', f'{img}.jpg'))
        plt.imshow(x)
        plt.show()
        print(image)
        print("Caption:", caption)
        print("_____________________________________")
        return caption


    def __init__(self):
        self.loader = Loader()
        self.model = Model()
        
        self.loader.load(self.model)

        self.model.build(self.loader)
        self.model.train()

if __name__ == '__main__':
    predictor = Predictor()
    predictor.generateCaptions()