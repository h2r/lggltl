import re
import os
import torch
import bcolz
import pickle
import numpy as np


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    src_lines = open(lang1, 'rb').read().strip().split('\n')
    tar_lines = open(lang2, 'rb').read().strip().split('\n')
    assert len(src_lines) == len(tar_lines)

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s), t] for s, t in zip(src_lines, tar_lines)]
    max_len = max([len(p[0]) for p in pairs])
    max_tar_len = max([len(p[1]) for p in pairs])

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs, max_len, max_tar_len


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs, max_len, max_tar_len = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs, max_len, max_tar_len


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(path) # self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(path) # self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(path) # self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


def vectors_for_input_language(lang):
    # source for this code: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
    glove_path = '/home/ng/workspace/lggltl/lggltl/glove.6B/'
    vectors = bcolz.open(glove_path + '6B.50.dat')[:]
    words = pickle.load(open(glove_path + '6B.50_words.pkl', 'rb'))
    word2idx = pickle.load(open(glove_path + '6B.50_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}

    target_vocab = lang.index2word

    emb_dim = 50

    matrix_len = len(target_vocab)
    weights_matrix = np.zeros((matrix_len, emb_dim))
    words_found = 0
    i=0

    for word in target_vocab:
        try:
            # print(target_vocab[word])
            weights_matrix[i] = glove[target_vocab[word]]
            # print(i)
            # print(word)
            words_found += 1
            i+=1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim,))

    return weights_matrix
