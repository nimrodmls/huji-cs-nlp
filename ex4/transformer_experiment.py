import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from exercise_blanks import device, test_model, plot_results
import data_loader
import numpy as np

TOKENIZED = "tokenized"
TRAIN = "train"
VAL = "val"
TEST = "test"

class Dataset(torch.utils.data.Dataset):
    """
    Dataset for loading data
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, 
                 dataset_path="stanfordSentimentTreebank", 
                 batch_size=50):
        """
        builds the data manager used for training and evaluation.
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=False)
        # map data splits to sentences lists
        self.sentences = {}
        self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()
        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        self.sentences_p = {}
        self.sentences_p[TRAIN] = self._preprocess_sentences(self.sentences[TRAIN])
        self.sentences_p[VAL] = self._preprocess_sentences(self.sentences[VAL])
        self.sentences_p[TEST] = self._preprocess_sentences(self.sentences[TEST])

        self.tokenizer = AutoTokenizer.from_pretrained('distilroberta-base', add_prefix_space=True)

        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: Dataset(self.tokenizer(x, truncation=True, padding=True, is_split_into_words=True), y) for
                               k, (x, y) in self.sentences_p.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def _preprocess_sentences(self, sents):
        """
        """
        x = []
        y = []
        for sent in sents:
            x.append(sent.text)
            y.append(sent.sentiment_class)
        return x, y
    
    def get_rare_words_iterator(self):
        """
        """
        rare_indices = data_loader.get_rare_words_examples(self.sentences[TEST], self.sentiment_dataset)
        subset_sents = self._preprocess_sentences([self.sentences[TEST][i] for i in rare_indices])
        x, y = subset_sents
        dataset = Dataset(self.tokenizer(x, truncation=True, padding=True, is_split_into_words=True), y)
        iterator = DataLoader(dataset, batch_size=64, shuffle=False)
        return iterator
    
    def get_negated_polarity_iterator(self):
        """
        """
        negated_indices = data_loader.get_negated_polarity_examples(self.sentences[TEST])
        subset_sents = self._preprocess_sentences([self.sentences[TEST][i] for i in negated_indices])
        x, y = subset_sents
        dataset = Dataset(self.tokenizer(x, truncation=True, padding=True, is_split_into_words=True), y)
        iterator = DataLoader(dataset, batch_size=64, shuffle=False)
        return iterator

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


def transformer_classification():

    def train_epoch(model, data_iterator, optimizer, criterion):
        """
        Perform an epoch of training of the model with the optimizer
        :param model:
        :param data_loader:
        :param optimizer:
        :return: Average loss over the epoch
        """
        model.train()
        total_loss = 0.
        pred_correct = 0
        # iterate over batches
        for batch in tqdm(data_iterator):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze()
            loss = criterion(logits, labels.to(torch.float32))
            loss.backward()
            optimizer.step()

            preds = data_loader.get_sentiment_class_from_logits(logits)
            pred_correct += (preds == labels).sum().item()
            total_loss += loss.item()
        
        print(f"TRAIN Accuracy: {pred_correct / len(data_iterator.dataset)}")
        return total_loss / len(data_iterator)

    def evaluate_model(model, data_iterator):
        model.eval()
        pred_correct = 0
        for batch in tqdm(data_iterator):
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze()

                preds = data_loader.get_sentiment_class_from_logits(logits)
                pred_correct += (preds == labels).sum().item()

        print(f"VAL Accuracy: {pred_correct / len(data_iterator.dataset)}")

    # Parameters
    epochs = 2
    batch_size = 64
    learning_rate = 1e-5

    # Model, tokenizer, and metric
    model = AutoModelForSequenceClassification.from_pretrained(
        'distilroberta-base', num_labels=1).to(device)
    dm = DataManager(batch_size=batch_size)

    # Weight decay is defaulted to 0 (as the exercise requires)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)

    val_accuracy = []
    train_loss = []

    for epoch in range(epochs):
        print(f"[TRAIN] Epoch: {epoch + 1}")

        loss = train_epoch(model, dm.get_torch_iterator(), optimizer, criterion)
        train_loss.append(loss)

        evaluate_model(model, dm.get_torch_iterator(data_subset=VAL))
        # val_accuracy.append(metric.compute()['accuracy'])

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

    for i, batch in enumerate(tqdm(data_iterator)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
                
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        outputs = outputs.logits.squeeze()
        loss = criterion(outputs, labels.to(torch.float32))
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
        
        for i, batch in enumerate(tqdm(data_iterator)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            outputs = outputs.logits.squeeze()

            # Calculate loss if criterion is given - It is not mandatory 
            loss = 0
            if None != criterion:
                loss = criterion(outputs, labels.to(torch.float32))

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
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    
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
    iterator = data_manager.get_negated_polarity_iterator()
    _, test_accuracy = evaluate(model, iterator)
    print(f"Negated Polarity Test Accuracy: {test_accuracy:.4f}")

    # Testing the rare words subset
    iterator = data_manager.get_rare_words_iterator()
    _, test_accuracy = evaluate(model, iterator)
    print(f"Rare Words Test Accuracy: {test_accuracy:.4f}")


def transformer_experiment(load_pretrained=False):
    """
    """
    # Parameters
    epochs = 2
    batch_size = 64
    learning_rate = 1e-5

    dm = DataManager(batch_size=batch_size)
    model = None
    if load_pretrained:
        model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base', num_labels=1).to(device)
        model.load_state_dict(torch.load('transformer_model.pth'))
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            'distilroberta-base', num_labels=1).to(device)
        train_accuracies, val_accuracies, train_losses, val_losses = \
            train_model(model, dm, epochs, learning_rate)
        torch.save(model.state_dict(), 'transformer_model.pth')
        plot_results('transformer', train_accuracies, val_accuracies, train_losses, val_losses)

    test_model(model, dm)

# transformer_classification()
transformer_experiment(load_pretrained=True)