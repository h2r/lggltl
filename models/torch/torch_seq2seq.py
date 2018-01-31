from __future__ import print_function, division
import sys

from lang import *
from networks import *
from train_eval import *


use_cuda = torch.cuda.is_available()

src, tar = '../../data/hard_pc_src_syn.txt', '../../data/hard_pc_tar_syn.txt'
# src, tar = '../../data/hard_pc_src.txt', '../../data/hard_pc_tar.txt'

SEED = int(sys.argv[1])
MODE = int(sys.argv[2])
random.seed(SEED)
torch.manual_seed(SEED)
print('Running with random seed {0}'.format(SEED))

input_lang, output_lang, pairs, MAX_LENGTH, MAX_TAR_LENGTH = prepareData(src, tar, False)
random.shuffle(pairs)
print('Maximum source sentence length: {0}'.format(MAX_LENGTH))
print(random.choice(pairs))

embed_size = 50
hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, embed_size, hidden_size)
attn_decoder1 = AttnDecoderRNN(embed_size, hidden_size, output_lang.n_words)
new_attn_decoder1 = NewAttnDecoderRNN(embed_size, hidden_size, output_lang.n_words, MAX_LENGTH)
decoder1 = DecoderRNN(embed_size, hidden_size, output_lang.n_words)

if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()
    new_attn_decoder1 = new_attn_decoder1.cuda()
    decoder1 = decoder1.cuda()

SAVE = False
CLI = False


def main():
    if MODE == 0:
        trainIters(input_lang, output_lang, encoder1, attn_decoder1, pairs, 10000, MAX_LENGTH, print_every=500)
        encoder1.eval()
        attn_decoder1.eval()
        evaluateRandomly(input_lang, output_lang, encoder1, attn_decoder1, pairs, MAX_LENGTH)
    elif MODE == 1:
        trainIters(input_lang, output_lang, encoder1, attn_decoder1, pairs, 10000, MAX_LENGTH, print_every=500)
        encoder1.eval()
        attn_decoder1.eval()
        evaluateTraining(input_lang, output_lang, encoder1, attn_decoder1, pairs, MAX_LENGTH)
    elif MODE == 2:
        print('Running cross validation on encoder and BA decoder...')
        crossValidation(input_lang, output_lang, encoder1, attn_decoder1, pairs, MAX_LENGTH)
    elif MODE == 3:
        print('Running cross validation on encoder and vanilla decoder...')
        crossValidation(input_lang, output_lang, encoder1, decoder1, pairs, MAX_LENGTH)
    elif MODE == 4:
        print('Running cross validation on encoder and EAA decoder...')
        crossValidation(input_lang, output_lang, encoder1, new_attn_decoder1, pairs, MAX_LENGTH)
    elif MODE == 5:
        print('Running generalization experiment with encoder and BA decoder...')
        results = []
        for i in range(1, 10):
            acc = evalGeneralization(input_lang, output_lang, encoder1, attn_decoder1, pairs, 0.1 * i, MAX_LENGTH)
            results.append(acc)
            encoder1.apply(resetWeights)
            attn_decoder1.apply(resetWeights)
        print(', '.join(map(str, results)))
    elif MODE == 6:
        print('Running generalization experiment with encoder and EAA decoder...')
        results = []
        for i in range(1, 10):
            acc = evalGeneralization(input_lang, output_lang, encoder1, new_attn_decoder1, pairs, 0.1 * i, MAX_LENGTH)
            results.append(acc)
            encoder1.apply(resetWeights)
            attn_decoder1.apply(resetWeights)
        print(', '.join(map(str, results)))
    elif MODE == 7:
        print('Running generalization experiment with encoder and vanilla decoder...')
        results = []
        for i in range(1, 10):
            acc = evalGeneralization(input_lang, output_lang, encoder1, decoder1, pairs, 0.1 * i, MAX_LENGTH)
            results.append(acc)
            encoder1.apply(resetWeights)
            attn_decoder1.apply(resetWeights)
        print(', '.join(map(str, results)))
    # elif MODE == 7:
    #     results = []
    #     for i in range(1, 10):
    #         acc = evalSampleEff(input_lang, output_lang, encoder1, attn_decoder1, pairs, 0.1 * i, MAX_LENGTH)
    #         results.append(acc)
    #         encoder1.apply(resetWeights)
    #         attn_decoder1.apply(resetWeights)
    #     print(', '.join(map(str, results)))
    else:
        print('Unknown MODE specified...exiting...')
        sys.exit(0)

    if SAVE:
        print('Serializing trained model...')
        torch.save(encoder1, './pytorch_encoder')
        torch.save(attn_decoder1, './pytorch_decoder')
        print('Serialized trained model to disk...')

    if CLI:
        while True:
            try:
                input_sentence = raw_input("Enter a command: ")
                output_words, attentions = evaluate(input_lang, output_lang, encoder1, attn_decoder1, input_sentence, MAX_LENGTH)
                print('input =', input_sentence)
                print('output =', ' '.join(output_words))
            except EOFError:
                break

main()