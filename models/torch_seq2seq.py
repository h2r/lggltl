from __future__ import print_function, division
from io import open
import re
import random
import itertools

import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import time
import math


use_cuda = torch.cuda.is_available()

SOS_token = 0
EOS_token = 1
UNK_token = 2

src = '../data/hard_pc_src_syn.txt'
tar = '../data/hard_pc_tar_syn.txt'

SEED = 1234 if len(sys.argv) != 2 else int(sys.argv[1])
random.seed(SEED)
torch.manual_seed(SEED)
print('Running with random seed {0}'.format(SEED))


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


MAX_LENGTH = 200


def readLangs(lang1, lang2, reverse=False):
    global MAX_LENGTH
    print("Reading lines...")

    # Read the file and split into lines
    src_lines = open(lang1, 'rb').read().strip().split('\n')
    tar_lines = open(lang2, 'rb').read().strip().split('\n')
    assert len(src_lines) == len(tar_lines)

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s), t] for s, t in zip(src_lines, tar_lines)]
    MAX_LENGTH = max([len(p[0]) for p in pairs])

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData(src, tar, False)
random.shuffle(pairs)
print(random.choice(pairs))


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, dropout_p=0.5):
        super(EncoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(input_size, embed_size)
        self.dropout1 = nn.Dropout(self.dropout_p)
        self.dropout2 = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(embed_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = self.dropout1(embedded)
        output, hidden = self.gru(output, hidden)
        output = self.dropout2(output)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, dropout_p=0.5):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.dropout_p = dropout_p
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(self.embed_size, self.hidden_size)
        self.dropout1 = nn.Dropout(self.dropout_p)
        self.dropout2 = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, encoder_outputs):
        output = self.embedding(input).view(1, 1, -1)
        output = self.dropout1(output)
        output, hidden = self.gru(output, hidden)
        output = self.dropout2(output)
        output = self.softmax(self.out(output[0]))
        return output, hidden, None

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class AttnDecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, dropout_p=0.5, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.embed_size)
        self.attn = nn.Linear(self.embed_size + self.hidden_size, self.max_length)
        self.attn_combine = nn.Linear(self.embed_size + self.hidden_size, self.hidden_size)
        self.dropout1 = nn.Dropout(self.dropout_p)
        self.dropout2 = nn.Dropout(self.dropout_p)
        self.dropout3 = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout1(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output = self.dropout2(output)
        output, hidden = self.gru(output, hidden)
        output = self.dropout3(output)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromPair(pair):
    input_variable = variableFromSentence(input_lang, ' '.join(list(reversed(pair[0].split()))))
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)


teacher_forcing_ratio = 0.5


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder, samples, n_iters, print_every=1000, plot_every=100, learning_rate=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    training_pairs = itertools.cycle(iter([variablesFromPair(s) for s in samples]))
    criterion = nn.NLLLoss()

    for i in range(1, n_iters + 1):
        training_pair = training_pairs.next()
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if i % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, i / n_iters),
                                         i, i / n_iters * 100, print_loss_avg))

        if i % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    input_variable = variableFromSentence(input_lang, ' '.join(list(reversed(sentence.split()))))
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        if decoder_attention is not None:
            decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def evaluateTraining(encoder, decoder):
    corr, tot = 0, 0
    for p in pairs:
        output_words, attentions = evaluate(encoder, decoder, p[0])
        output_words = ' '.join(output_words[:-1])
        if output_words == p[1]:
            corr += 1
        # print('Input: {0}\tOutput: {1}\tExpected:{2}'.format(p[0], output_words, p[1]))
        tot += 1

    print('Training Accuracy: {0}/{1} = {2}%'.format(corr, tot, corr / tot))


def evaluateSamples(encoder, decoder, samples):
    corr, tot = 0, 0
    for p in samples:
        output_words, attentions = evaluate(encoder, decoder, p[0])
        output_words = ' '.join(output_words[:-1])
        if output_words == p[1]:
            corr += 1
        tot += 1
    return corr, tot, corr / tot


def resetWeights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def crossValidation(encoder, decoder, samples, n_folds=5):
    for _ in range(10):
        random.shuffle(samples)

    correct, total = 0, 0
    fold_range = range(0, len(samples), int(len(samples) / n_folds))
    fold_range.append(len(samples))

    print('Starting {0}-fold cross validation'.format(n_folds))
    for f in xrange(n_folds):
        print('Running cross validation fold {0}/{1}...'.format(f + 1, n_folds))

        train_samples = samples[:fold_range[f]] + samples[fold_range[f + 1]:]
        val_samples = samples[fold_range[f]:fold_range[f + 1]]

        trainIters(encoder, decoder, train_samples, 10000, print_every=500)

        encoder1.eval()
        attn_decoder1.eval()

        corr, tot, acc = evaluateSamples(encoder, decoder, val_samples)
        print('Cross validation fold #{0} Accuracy: {1}/{2} = {3}%'.format(f + 1, corr, tot, 100. * acc))
        correct += corr
        total += tot

        encoder.apply(resetWeights)
        decoder.apply(resetWeights)
    print('{0}-fold Cross Validation Accuracy: {1}/{2} = {3}%'.format(n_folds, correct, total, 100. * correct / total))


def evalGeneralization(encoder, decoder, samples, perc):
    for _ in range(10):
        random.shuffle(samples)
    
    tar_set = list(set([s[1] for s in samples]))
    tar_num = int(np.ceil(perc * len(tar_set)))
    train_forms = random.sample(tar_set, tar_num)
    print('GLTL Training Formulas: {0}'.format(train_forms))
    print('GLTL Evaluation Formulas: {0}'.format([s for s in tar_set if s not in train_forms]))
    train_samples = [s for s in samples if s[1] in train_forms]
    eval_samples = [s for s in samples if s[1] not in train_forms]

    print('Training with {0}/{3} unique GLTL formulas => {1} training samples | {2} testing samples'.format(tar_num, len(train_samples), len(eval_samples), len(tar_set)))
    trainIters(encoder, decoder, train_samples, 10000, print_every=10000)

    encoder1.eval()
    attn_decoder1.eval()

    corr, tot, acc = evaluateSamples(encoder, decoder, eval_samples)
    print('Held-out Accuracy: {0}/{1} = {2}%'.format(corr, tot, 100. * acc))
    return acc


def evalSampleEff(encoder, decoder, samples, perc):
    for _ in range(10):
        random.shuffle(samples)

    train_samples = random.sample(samples, int(perc * len(samples)))
    eval_samples = [s for s in samples if s not in train_samples]
    print('Training with {0}/{1} random data samples'.format(len(train_samples), len(samples)))
    trainIters(encoder, decoder, train_samples, 10000, print_every=10000)

    encoder1.eval()
    attn_decoder1.eval()

    corr, tot, acc = evaluateSamples(encoder, decoder, eval_samples)
    print('Held-out Accuracy: {0}/{1} = {2}%'.format(corr, tot, 100. * acc))
    return acc


embed_size = 50
hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, embed_size, hidden_size)
attn_decoder1 = AttnDecoderRNN(embed_size, hidden_size, output_lang.n_words, dropout_p=0.5)
decoder1 = DecoderRNN(embed_size, hidden_size, output_lang.n_words)

if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()
    decoder1 = decoder1.cuda()

# print('Beginning training...')
# trainIters(encoder1, attn_decoder1, pairs, 10000, print_every=500)
# encoder1.eval()
# attn_decoder1.eval()
# evaluateRandomly(encoder1, attn_decoder1)
# evaluateTraining(encoder1, attn_decoder1)
# crossValidation(encoder1, attn_decoder1, pairs)
# crossValidation(encoder1, decoder1, pairs)
results = []
for i in range(1, 10):
    # acc = evalGeneralization(encoder1, attn_decoder1, pairs, 0.1 * i)
    acc = evalSampleEff(encoder1, attn_decoder1, pairs, 0.1 * i)
    results.append(acc)
    encoder1.apply(resetWeights)
    attn_decoder1.apply(resetWeights)
print(','.join(map(str, results)))

# print('Serializing trained model...')
# torch.save(encoder1, './pytorch_encoder')
# torch.save(attn_decoder1, './pytorch_decoder')
# print('Serialized trained model to disk...')

'''
while True:
    try:
        input_sentence = raw_input("Enter a command: ")
        output_words, attentions = evaluate(encoder1, attn_decoder1, input_sentence)
        print('input =', input_sentence)
        print('output =', ' '.join(output_words))
    except EOFError:
        break
'''
