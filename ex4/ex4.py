import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = get_available_device()

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.key_to_index.keys())
    print(wv_from_bin.key_to_index[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    embeddings = sentence_to_embedding(sent, word_to_vec, len(sent.text), embedding_dim)
    return torch.mean(embeddings, dim=0)


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    vec = torch.zeros(size=(size,))
    vec[ind] = 1
    return vec


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    vec = torch.zeros(size=(len(word_to_ind),))
    for word in sent.text:
        if word in word_to_ind:
            vec += get_one_hot(len(word_to_ind), word_to_ind[word])
    return vec / len(sent.text) # normalize by the number of words in the sentence


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    return {word: ind for ind, word in enumerate(words_list)}


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    embeddings = torch.zeros(size=(seq_len, embedding_dim))
    for idx, word in enumerate(sent.text[:seq_len]):
        if word in word_to_vec:
            embeddings[idx] = torch.from_numpy(word_to_vec[word])
        # else - if the word is not in the w2v dict, we leave the zero vector
    return embeddings


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank", batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list, cache_w2v=True),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list, cache_w2v=True),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape




# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    The LSTM is bidirectional. The model is implemented using PyTorch's nn.LSTM module.
    """
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super(LSTM, self).__init__()
        # LSTM Model - Note that it is bidirectional, and that batch_first is set to True
        # since this is the way we are passing the data into forward
        # (meaning the shape of the text is (batch_size, seq_len, embedding_dim))
        self.model = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers,
                             dropout=dropout, bidirectional=True, batch_first=True)
        # LSTM is bidirectional, so we multiply by 2 to support concatenation
        self.linear = nn.Linear(hidden_dim * 2, 1)

    def forward(self, text):
        output, (h_n, c_n) = self.model(text)
        return self.linear(torch.hstack([h_n[0], h_n[1]])) # Passing concatenated output


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """
    def __init__(self, embedding_dim):
        super(LogLinear, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, 1))

    def forward(self, x):
        return self.model(x)


# ------------------------- training functions -------------


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    :return: tuple of (loss, accuracy) for the epoch.
    """
    model.train()

    pred_correct = 0
    accumulated_loss = 0.

    for i, (inputs, labels) in enumerate(tqdm(data_iterator)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = data_loader.get_sentiment_class_from_logits(outputs)
        pred_correct += (preds == labels).sum().item()
        accumulated_loss += loss.item()

    ep_accuracy = pred_correct / len(data_iterator.dataset)
    ep_loss = accumulated_loss / len(data_iterator)

    return ep_loss, ep_accuracy


def evaluate(model, data_iterator, criterion=None):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    model.eval()
    with torch.no_grad():
        pred_correct = 0
        accumulated_loss = 0.
        
        for i, (inputs, labels) in enumerate(tqdm(data_iterator)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze()

            # Calculate loss if criterion is given - It is not mandatory 
            loss = 0
            if None != criterion:
                loss = criterion(outputs, labels)

            preds = data_loader.get_sentiment_class_from_logits(outputs)
            pred_correct += (preds == labels).sum().item()

            if None != criterion:
                accumulated_loss += loss.item()

        accuracy = pred_correct / len(data_iterator.dataset)
        loss = accumulated_loss / len(data_iterator)
        
        return loss, accuracy


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss().to(device)
    
    train_accuracies, val_accuracies = [], []
    train_losses, val_losses = [], []

    for ep in range(n_epochs):
        print(f"Epoch {ep+1}")
        train_loss, train_accuracy = train_epoch(model, data_manager.get_torch_iterator(), optimizer, criterion)
        val_loss, val_accuracy = evaluate(model, data_manager.get_torch_iterator(data_subset=VAL), criterion)
        print(f"Epoch {ep+1}: Train loss {train_loss:.4f}, Train acc {train_accuracy:.4f}, Val loss {val_loss:.4f}, Val acc {val_accuracy:.4f}")

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    return train_accuracies, val_accuracies, train_losses, val_losses


def test_model(model, data_manager):
    """
    Tests the given model with the regular subset & special subsets (rare words & negations)
    """
    # Testing the full subset
    _, test_accuracy = evaluate(model, data_manager.get_torch_iterator(data_subset=TEST))
    print(f"Full Subset Test Accuracy: {test_accuracy:.4f}")

    # Testing the negated polarity subset
    negated_indices = data_loader.get_negated_polarity_examples(data_manager.sentences[TEST])
    subset_sents = [data_manager.sentences[TEST][i] for i in negated_indices]
    iterator = DataLoader(OnlineDataset(subset_sents, data_manager.sent_func, data_manager.sent_func_kwargs),
                          batch_size=64, shuffle=False)
    _, test_accuracy = evaluate(model, iterator)
    print(f"Negated Polarity Test Accuracy: {test_accuracy:.4f}")

    # Testing the rare words subset
    rare_indices = data_loader.get_rare_words_examples(data_manager.sentences[TEST], data_manager.sentiment_dataset)
    subset_sents = [data_manager.sentences[TEST][i] for i in rare_indices]
    iterator = DataLoader(OnlineDataset(subset_sents, data_manager.sent_func, data_manager.sent_func_kwargs),
                          batch_size=64, shuffle=False)
    _, test_accuracy = evaluate(model, iterator)
    print(f"Rare Words Test Accuracy: {test_accuracy:.4f}")


def plot_results(experiment_name, train_accuracies, val_accuracies, train_losses, val_losses):
    """
    Plotting the training accuracies & validation accuracies on the same plot
    Plotting the training losses & validation losses on the same plot

    :param experiment_name: The name of the experiment, for saving the plots
    """
    epochs = range(1, len(train_accuracies) + 1) # Assuming the lengths are the same

    plt.figure()
    plt.plot(epochs, train_accuracies, label='Train')
    plt.plot(epochs, val_accuracies, label='Validation')
    plt.title('Accuracy')
    plt.xticks(epochs)
    plt.legend()
    plt.savefig(f'{experiment_name}_accuracy.png')

    plt.figure()
    plt.plot(epochs, train_losses, label='Train')
    plt.plot(epochs, val_losses, label='Validation')
    plt.title('Loss')
    plt.xticks(epochs)
    plt.legend()
    plt.savefig(f'{experiment_name}_loss.png')

def save_results(experiment_name, train_accuracies, val_accuracies, train_losses, val_losses):
    """
    Save the results of the experiment to a pickle file
    """
    results = {
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_losses': train_losses,
        'val_losses': val_losses
    }

    save_pickle(results, f'{experiment_name}_results.pkl')

def load_results(experiment_name):
    """
    Load the results of the experiment from a pickle file
    """
    return load_pickle(f'{experiment_name}_results.pkl')

def log_linear_with_one_hot(load_pretrained=False):
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    # Learning Parameters (as defined in the exercise)
    batch_size = 64
    lr = 0.01
    weight_decay = 0.001
    n_epochs = 20

    dm = DataManager(data_type=ONEHOT_AVERAGE, batch_size=batch_size)
    model = None
    if load_pretrained:
        model = LogLinear(dm.get_input_shape()[0])
        model.load_state_dict(torch.load('log_linear_one_hot.pth'))
        model.to(device)
    else:
        model = LogLinear(dm.get_input_shape()[0])
        model.to(device)
        train_accuracies, val_accuracies, train_losses, val_losses = \
            train_model(model, dm, n_epochs=n_epochs, lr=lr, weight_decay=weight_decay)
        torch.save(model.state_dict(), 'log_linear_one_hot.pth')

        save_results('log_linear_one_hot', train_accuracies, val_accuracies, train_losses, val_losses)
        plot_results('log_linear_one_hot', train_accuracies, val_accuracies, train_losses, val_losses)

    test_model(model, dm)
    

def log_linear_with_w2v(load_pretrained=False):
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    # Learning Parameters (as defined in the exercise)
    batch_size = 64
    lr = 0.01
    weight_decay = 0.001
    n_epochs = 20
    embedding_dim = 300

    dm = DataManager(data_type=W2V_AVERAGE, batch_size=batch_size, embedding_dim=embedding_dim)
    model = None
    if load_pretrained:
        model = LogLinear(dm.get_input_shape()[0])
        model.load_state_dict(torch.load('log_linear_w2v.pth'))
        model.to(device)
    else:
        model = LogLinear(dm.get_input_shape()[0])
        model.to(device)
        train_accuracies, val_accuracies, train_losses, val_losses = \
            train_model(model, dm, n_epochs=n_epochs, lr=lr, weight_decay=weight_decay)
        torch.save(model.state_dict(), 'log_linear_w2v.pth')
            
        save_results('log_linear_w2v', train_accuracies, val_accuracies, train_losses, val_losses)
        plot_results('log_linear_w2v', train_accuracies, val_accuracies, train_losses, val_losses)

    test_model(model, dm)


def lstm_with_w2v(load_pretrained=False):
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    # Learning Parameters (as defined in the exercise)
    batch_size = 64
    lr = 0.001
    weight_decay = 0.0001
    n_epochs = 4
    embedding_dim = 300

    # Hyperparameters for the LSTM model
    hidden_dim = 128
    n_layers = 1
    dropout = 0.5

    dm = DataManager(data_type=W2V_SEQUENCE, batch_size=batch_size, embedding_dim=embedding_dim)
    model = None
    if load_pretrained:
        model = LSTM(embedding_dim, hidden_dim, n_layers, dropout)
        model.load_state_dict(torch.load('lstm_w2v.pth'))
        model.to(device)
    else:
        model = LSTM(embedding_dim, hidden_dim, n_layers, dropout)
        model.to(device)
        train_accuracies, val_accuracies, train_losses, val_losses = \
            train_model(model, dm, n_epochs=n_epochs, lr=lr, weight_decay=weight_decay)
        torch.save(model.state_dict(), 'lstm_w2v.pth')

        save_results('lstm_w2v', train_accuracies, val_accuracies, train_losses, val_losses)
        plot_results('lstm_w2v', train_accuracies, val_accuracies, train_losses, val_losses)

    test_model(model, dm)

def plot_from_saved_results(experiment_name):
    """
    Plot the results of the experiment from the saved pickle file
    """
    results = load_results(experiment_name)
    plot_results(experiment_name, results['train_accuracies'], results['val_accuracies'], results['train_losses'], results['val_losses'])

if __name__ == '__main__':
    # Setting seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    log_linear_with_one_hot(load_pretrained=False)
    log_linear_with_w2v(load_pretrained=False)
    lstm_with_w2v(load_pretrained=False)

    # plot_from_saved_results('log_linear_one_hot')
    # plot_from_saved_results('log_linear_w2v')
    # plot_from_saved_results('lstm_w2v')