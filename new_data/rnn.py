import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

from sklearn import preprocessing
from collections import Counter
import numpy
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader


class Vocab(object):
    def __init__(self, iterable, max_size=None, sos_token=None, eos_token=None, unk_token='<unk>'):
        """Initialize the vocabulary.
        Args:
            iterable: An iterable which produces sequences of tokens used to update
                the vocabulary.
            max_size: (Optional) Maximum number of tokens in the vocabulary.
            sos_token: (Optional) Token denoting the start of a sequence.
            eos_token: (Optional) Token denoting the end of a sequence.
            unk_token: (Optional) Token denoting an unknown element in a
                sequence.
        """
        self.max_size = max_size
        self.pad_token = '<pad>'
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.stop_words = ['your a the is and or in be to of for not on with as by']

        # Add special tokens.
        id2word = [self.pad_token]
        if sos_token is not None:
            id2word.append(self.sos_token)
        if eos_token is not None:
            id2word.append(self.eos_token)
        if unk_token is not None:
            id2word.append(self.unk_token)

        # Update counter with token counts.
        counter = Counter()
        for x in iterable:
            if x not in self.stop_words:
                counter.update(x)

        # Extract lookup tables.
        if max_size is not None:
            counts = counter.most_common(max_size)
        else:
            counts = counter.items()
            counts = sorted(counts, key=lambda x: x[1], reverse=True)
        words = [x[0] for x in counts]
        id2word.extend(words)
        word2id = {x: i for i, x in enumerate(id2word)}

        self._id2word = id2word
        self._word2id = word2id

    def __len__(self):
        return len(self._id2word)

    def word2id(self, word):
        """Map a word in the vocabulary to its unique integer id.
        Args:
            word: Word to lookup.
        Returns:
            id: The integer id of the word being looked up.
        """
        if word in self._word2id:
            return self._word2id[word]
        elif self.unk_token is not None:
            return self._word2id[self.unk_token]
        else:
            raise KeyError('Word "%s" not in vocabulary.' % word)

    def id2word(self, id):
        """Map an integer id to its corresponding word in the vocabulary.
        Args:
            id: Integer id of the word being looked up.
        Returns:
            word: The corresponding word.
        """
        return self._id2word[id]


class Annotation(object):
    def __init__(self):
        """A helper object for storing annotation data."""
        self.tokens = []
        self.label = None


class LabeledDataset(Dataset):
    def __init__(self, x_data, y_data):
        """Initializes the SentimentDataset.
        Args:
            x_data: The list of loaded input field data
        """
        self.annotations = []
        for i, d in enumerate(x_data):
            annotation = Annotation()
            annotation.tokens = d['feature'].split()
            annotation.label = y_data[i]
            self.annotations.append(annotation)
        self.token_vocab = Vocab([x.tokens for x in self.annotations])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        token_ids = [self.token_vocab.word2id(x) for x in annotation.tokens]
        target = annotation.label
        return token_ids, target


def pad(sequences, max_length, pad_value=0):
    """Pads a list of sequences.
    Args:
        sequences: A list of sequences to be padded.
        max_length: The length to pad to.
        pad_value: The value used for padding.
    Returns:
        A list of padded sequences.
    """
    out = []
    for sequence in sequences:
        padded = sequence + [0]*(max_length - len(sequence))
        out.append(padded)
    return out


def collate_annotations(batch):
    """Function used to collate data returned by CoNLLDataset."""
    # Get inputs, targets, and lengths.
    inputs, targets = zip(*batch)
    lengths = [len(x) for x in inputs]
    # Sort by length.
    sort = sorted(zip(inputs, targets, lengths),
                  key=lambda x: x[2],
                  reverse=True)
    inputs, targets, lengths = zip(*sort)
    # Pad.
    max_length = max(lengths)
    inputs = pad(inputs, max_length)
    # Transpose.
    inputs = list(map(list, zip(*inputs)))
    # Convert to PyTorch variables.
    inputs = Variable(torch.LongTensor(inputs))
    targets = Variable(torch.LongTensor(targets))
    lengths = Variable(torch.LongTensor(lengths))
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        targets = targets.cuda()
        lengths = lengths.cuda()
    return inputs, targets, lengths


class SentimentClassifier(nn.Module):
    def __init__(self,
                 input_vocab_size,
                 output_vocab_size,
                 embedding_dim=64,
                 hidden_size=64):
        """Initializes the tagger.

        Args:
            input_vocab_size: Size of the input vocabulary.
            output_vocab_size: Size of the output vocabulary.
            embedding_dim: Dimension of the word embeddings.
            hidden_size: Number of units in each LSTM hidden layer.
        """
        # Always do this!!!
        super(SentimentClassifier, self).__init__()

        # Store parameters
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # Define layers
        self.word_embeddings = nn.Embedding(input_vocab_size, embedding_dim,
                                            padding_idx=0)
        self.rnn = nn.GRU(embedding_dim, hidden_size)  # , dropout=0.9)
        self.fc = nn.Linear(hidden_size, output_vocab_size)
        self.activation = nn.LogSoftmax(dim=2)

    def forward(self, x, lengths=None, hidden=None):
        """Computes a forward pass of the language model.

        Args:
            x: A LongTensor w/ dimension [seq_len, batch_size].
            lengths: The lengths of the sequences in x.
            hidden: Hidden state to be fed into the lstm.

        Returns:
            net: Probability of the next word in the sequence.
            hidden: Hidden state of the lstm.
        """
        seq_len, batch_size = x.size()

        # If no hidden state is provided, then default to zeros.
        if hidden is None:
            hidden = Variable(torch.zeros(1, batch_size, self.hidden_size))
            if torch.cuda.is_available():
                hidden = hidden.cuda()

        net = self.word_embeddings(x)
        if lengths is not None:
            lengths_list = lengths.data.view(-1).tolist()
            net = pack_padded_sequence(net, lengths_list)
        net, hidden = self.rnn(net, hidden)
        # NOTE: we are using hidden as the input to the fully-connected layer, not net!!!
        net = self.fc(hidden)
        net = self.activation(net)

        return net, hidden


def train(x_data, y_data):
    # Convert labels to integers
    le = preprocessing.LabelEncoder()
    le.fit(y_data)
    y_data = le.transform(y_data)

    # Load dataset.
    dataset = LabeledDataset(x_data, y_data)

    # Hyperparameters / constants.
    input_vocab_size = len(dataset.token_vocab)
    output_vocab_size = len(list(le.classes_))
    batch_size = int(len(x_data) / 300)  # for about 3000 iterations with 10 epochs
    epochs = 10

    # Initialize the model.
    model = SentimentClassifier(input_vocab_size, output_vocab_size)
    if torch.cuda.is_available():
        model = model.cuda()

    # Initialize loss function and optimizer.
    loss_function = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Main training loop.
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=collate_annotations)
    losses = []
    i = 0
    for epoch in range(epochs):
        for inputs, targets, lengths in data_loader:
            optimizer.zero_grad()
            outputs, _ = model(inputs, lengths=lengths)

            outputs = outputs.view(-1, output_vocab_size)
            targets = targets.view(-1)

            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            losses.append(loss.data[0])
            if (i % 100) == 0:
                average_loss = numpy.mean(losses)
                losses = []
                print('Iteration %i - Loss: %0.6f' % (i, average_loss))
            #if (i % 1000) == 0:
            #    torch.save(model, 'sentiment_classifier.pt')
            i += 1
    #torch.save(model, 'sentiment_classifier.final.pt')
    return model, dataset.token_vocab, le


def pred(model, token_vocab, label_encoder, x_test):
    y_pred = []
    for d in x_test:
        ids = [[token_vocab.word2id(x)] for x in d['feature'].split()]
        ids = Variable(torch.LongTensor(ids))
        if torch.cuda.is_available():
            ids = ids.cuda()
        # Get model output.
        output, _ = model(ids)
        _, pred = torch.max(output, dim=2)
        if torch.cuda.is_available():
            pred = pred.cpu()
        pred = pred.data.view(-1).numpy()
        y_pred.append(label_encoder.inverse_transform(pred)[0])
    return y_pred
