import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
import evaluate
from tqdm import tqdm
from exercise_blanks import device
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

        self.sentences[TRAIN] = self._preprocess_sentences(self.sentences[TRAIN])
        self.sentences[VAL] = self._preprocess_sentences(self.sentences[VAL])
        self.sentences[TEST] = self._preprocess_sentences(self.sentences[TEST])

        self.tokenizer = AutoTokenizer.from_pretrained('distilroberta-base', add_prefix_space=True)

        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: Dataset(self.tokenizer(x, truncation=True, padding=True, is_split_into_words=True), y) for
                               k, (x, y) in self.sentences.items()}
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
    epochs = 3
    batch_size = 64
    learning_rate = 5e-5

    # Model, tokenizer, and metric
    model = AutoModelForSequenceClassification.from_pretrained(
        'distilroberta-base', num_labels=1).to(device)
    dm = DataManager(batch_size=batch_size)

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

transformer_classification()