import torch
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import torch.functional as F
import torch.nn.functional as F


corpus = [
        'he is a king',
        'she is a queen',
        'he is a man',
        'she is a woman',
        'warsaw is poland capital',
        'berlin is germany capital',
        'paris is france capital',
]


def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens


tokenized_corpus = tokenize_corpus(corpus)

vocabulary = []
for sentence in tokenized_corpus:
    for token in sentence:
        if token not in vocabulary:
            vocabulary.append(token)

word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}
vocabulary_size = len(vocabulary)

window_size = 3
word_pairs = []
idx_pairs = []

# for each sentence
for sentence in tokenized_corpus:
    indices = [word2idx[word] for word in sentence]

    # for each word, threated as center word
    for center_word_pos in range(len(indices)):
        # for each window position
        for w in range(-window_size, window_size+1):
            context_word_pos = center_word_pos + w

            # make score not jump out sentence
            if context_word_pos < 0 or context_word_pos >= len(
                    indices) or center_word_pos == context_word_pos:
                continue

            context_word_idx = indices[context_word_pos]
            word_pairs.append(
                (sentence[center_word_pos],
                 sentence[context_word_pos]))
            idx_pairs.append((indices[center_word_pos], context_word_idx))

word_paris = np.array(word_pairs)
idx_pairs = np.array(idx_pairs)


def get_input_layer(word2idx):
    x = torch.zeros(vocabulary_size).float()
    x[word2idx] = 1.0
    return x


embedding_dims = 5
W1 = Variable(
    torch.randn(
        embedding_dims,
        vocabulary_size).float(),
    requires_grad=True)

W2 = Variable(
    torch.randn(
        vocabulary_size,
        embedding_dims).float(),
    requires_grad=True)

EPOCHS = 101
LEARNING_RATE = 0.001

for epoch in tqdm(range(EPOCHS)):
    loss_val = 0

    for data, target in idx_pairs:
        x = Variable(get_input_layer(data)).float()
        y_true = Variable(torch.from_numpy(np.array([target])).long())

        z1 = torch.matmul(W1, x)
        z2 = torch.matmul(W2, z1)

        log_softmax = F.log_softmax(z2, dim=0)

        loss = F.nll_loss(log_softmax.view(1, -1), y_true)
        loss_val += loss.data[0]
        loss.backward()
        W1.data -= LEARNING_RATE * W1.grad.data
        W2.data -= LEARNING_RATE * W2.grad.data

        W1.grad.data.zero_()
        W2.grad.data.zero_()

    if epoch % 10 == 0:
        tqdm.write('Loss at {0} : {1}'.format(epoch, loss_val/len(idx_pairs)))
