import numpy as np
import bcolz
import pickle

words = []
idx = 0
word2idx = {}
glove_path = '/home/ng/workspace/lggltl/lggltl/glove.6B/glove.6B.50d.txt'
vectors = bcolz.carray(np.zeros(1), rootdir='/home/ng/workspace/lggltl/lggltl/glove.6B/6B.50.dat', mode='w')

with open('/home/ng/workspace/lggltl/lggltl/glove.6B/glove.6B.50d.txt', 'rb') as f:
    for l in f:
        # print(l)
        line = l.decode('utf-8').split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)


vectors = bcolz.carray(vectors[1:].reshape((-1, 50)), rootdir='/home/ng/workspace/lggltl/lggltl/glove.6B/6B.50.dat', mode='w')
vectors.flush()
pickle.dump(words, open('/home/ng/workspace/lggltl/lggltl/glove.6B/6B.50_words.pkl', 'wb'))
pickle.dump(word2idx, open('/home/ng/workspace/lggltl/lggltl/glove.6B/6B.50_idx.pkl', 'wb'))
