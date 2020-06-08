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
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import Input, layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import csv

class Loader():
    root_path = None
    lookup = dict()
    lex = set()  # set of unique words
    train_descriptions = None
    all_train_captions = None
    max_length = None
    START = "startseq"
    STOP = "endseq"
    vocab_size = None

    encoding_train = None
    vocab = None
    idxtoword = {}
    wordtoidx = {}

    embedding_dim = 200
    embedding_matrix = None

    def __init__(self):
        self.load_root()

        # generate lookup
        memes_path = os.path.join(self.root_path, 'ImgFlip500K_Dataset', 'memes')
        self.load_memes(memes_path)
        # self.load_memes(memes_path, 0, 1)

        self.load_lex()

        self.generate_train_descriptions()# lookup with start/stop
        self.generate_all_train_descriptions()
        self.generate_vocab() # remove words with occurences < 1

        self.build_idtoword()

    def load(self, model_obj):
        train_path = os.path.join(self.root_path,'ImgFlip500K_Dataset',"data",f'train{model_obj.OUTPUT_DIM}.pkl')
        train_images_path = os.path.join(self.root_path,'ImgFlip500K_Dataset','templates','img') 
        self.generate_encoding_train(train_path, train_images_path, model_obj)

        glove_dir = os.path.join(self.root_path,'glove.6B')
        self.load_glove(glove_dir)

    def load_root(self):
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=True)
            print('Note: using Google CoLab')
            COLAB = True
        except:
            print('Note: not using Google Colab')
            COLAB = False

        if COLAB:
            self.root_path = "/content/drive/Shared drives/ai-memes"
        else:
            self.root_path = "./data/"

    def _clean_data(self, list):
        null_punct = str.maketrans('', '', string.punctuation)
        for desc in list:
            desc = [word.lower() for word in desc.split()]
            desc = [w.translate(null_punct) for w in desc]
            desc = [word for word in desc if word.isalpha()]
            desc = [word for word in desc if len(word) > 1]
            yield ' '.join(desc)

    def _isascii_word(self, w): return len(w) == len(w.encode())


    def _isascii_list(self, l): return reduce(lambda rez, word: rez &
                                    True if self._isascii_word(word) else False & rez, l, True)


    def load_memes(self, dirname, start=0, end=100):
        i = 0
        for filename in tqdm(os.listdir(dirname)):  # foreach json file in memes
            i += 1
            if i < start or i > end:
                continue
            meme_name = Path(filename).stem  # remove file extension
            with open(os.path.join(dirname, filename)) as json_file:
                memes = json.load(json_file)
                # lookup[meme_name] = list(map(lambda meme: ' | '.join(meme['boxes']), memes))
                self.lookup[meme_name] = []
                for meme in memes:
                    words = ' | '.join(self._clean_data(meme['boxes']))
                    if self._isascii_list(words):
                        self.lookup[meme_name].append(words)                                    
        print(f'Memes loaded: {len(self.lookup)}')  # 99 memes, in the latest dataset 100
        print(f'Meme example: {self.lookup[list(self.lookup)[0]][0]}') # when you off the dope | and you think you a bird

    def load_lex(self):
        # load unique words
        self.max_length = 0
        for desc in self.lookup:
            for word in desc.split():
                self.lex.add(word)
                if self.max_length < len(word):
                    self.max_length = max(self.max_length, len(word))
                    max_word = word

        print(f'Unique words: {len(self.lex)}')
        print(f'Biggest word: {self.max_length} chars')
        print(max_word)

    def generate_train_descriptions(self):
        # same as lookup but captions wrapped in Start/Stop
        self.train_descriptions = {k:v for k,v in self.lookup.items()}
        for _,v in self.train_descriptions.items(): 
            for d in range(len(v)):
                v[d] = f'{self.START} {v[d]} {self.STOP}'
        self.max_length += 2 # update maximum caption length
        print(f'Wrapped captions in Start/Stop')

    def generate_all_train_descriptions(self):
        # build all captions list
        self.all_train_captions = []
        for _, val in self.train_descriptions.items():
            for cap in val:
                self.all_train_captions.append(cap)

    def generate_vocab(self):
        # remove words with occurences < 1
        word_count_threshold = 1
        word_counts = {}
        nsents = 0 # number of sentences
        for sent in self.all_train_captions:
            nsents += 1
            for w in sent.split(' '):
                word_counts[w] = word_counts.get(w, 0) + 1

        self.vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
        print('preprocessed words %d ==> %d' % (len(word_counts), len(self.vocab)))

    # Nicely formatted time string
    def hms_string(self, sec_elapsed):
        h = int(sec_elapsed / (60 * 60))
        m = int((sec_elapsed % (60 * 60)) / 60)
        s = sec_elapsed % 60
        return f"{h}:{m:>02}:{s:>05.2f}"

    def generate_encoding_train(self, train_path, train_images_path, model_obj):
        # generate train set
        if not os.path.exists(train_path):
            start = time()
            self.encoding_train = {}
            for image_path in tqdm(os.listdir(train_images_path)):
                # print(image_path)
                img = tensorflow.keras.preprocessing.image.load_img(os.path.join(train_images_path,image_path), target_size=(model_obj.HEIGHT, model_obj.WIDTH))
                self.encoding_train[image_path] = model_obj.encodeImage(img)
            with open(train_path, "wb") as fp:
                pickle.dump(self.encoding_train, fp)
            print(f"\nGenerating training set took: {self.hms_string(time()-start)}")
        else:
            with open(train_path, "rb") as fp:
                self.encoding_train = pickle.load(fp)
        print('Loaded')

    def build_idtoword(self):
        # idxtoword and wordtoidx
        ix = 1
        for w in self.vocab:
            self.wordtoidx[w] = ix
            self.idxtoword[ix] = w
            ix += 1
            
        self.vocab_size = len(self.idxtoword) + 1 
        print(f'Vocab Size: {self.vocab_size}')

    def load_glove(self, glove_dir):
        embeddings_index = {} 
        f = open(os.path.join(glove_dir, 'glove.6B.200d.txt'), encoding="utf-8")

        for line in tqdm(f):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

        f.close()
        print(f'Found {len(embeddings_index)} word vectors.')

        # Get 200-dim dense vector for each of the 10000 words in out vocabulary
        self.embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim))
        not_found = []
        for word, i in self.wordtoidx.items():
            #if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in the embedding index will be all zeros
                self.embedding_matrix[i] = embedding_vector
            else: 
                not_found.append(word)
        print(f'Not founded words: {len(not_found)} from total of {len(self.wordtoidx)}')

        self.eliminated = 0
        for meme in tqdm(self.train_descriptions):
            self.train_descriptions[meme] = [desc for desc in self.train_descriptions[meme] if self._hasNotFoundWord(desc, not_found) == False ]
        print(f'Removed {self.eliminated}')

    def _hasNotFoundWord(self, desc, not_found):
        # delete wrong sentences
        stripped_desc = desc.split()[1:-1]
        if bool(set(not_found) & set(stripped_desc)):
            self.eliminated += 1
            return True
        else:
            return False

    def get_meme_name(self, id):
        return 'Yo Dawg Heard You'
        # csv_filename = os.path.join(self.root_path, 'ImgFlip500K_Dataset', 'popular_100_memes.csv')
        # print(csv_filename)
        # print(id)
        # with open(csv_filename) as csv_file:
        #     reader = csv.reader(csv_file, delimiter=',')
        #     line = 0
        #     for row in reader:
        #         line += 1
        #         if line == 1:
        #             print(f'HEADERS {row}')
        #         else:
        #             if row[0] == id:
        #                 return row[1]