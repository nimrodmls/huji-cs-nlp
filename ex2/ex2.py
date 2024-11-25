

###################################################
# Exercise 2 - Natural Language Processing 67658  #
###################################################

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# subset of categories that we will use
category_dict = {'comp.graphics': 'computer graphics',
                 'rec.sport.baseball': 'baseball',
                 'sci.electronics': 'science, electronics',
                 'talk.politics.guns': 'politics, guns'
                 }

def get_data(categories=None, portion=1.):
    """
    Get data for given categories and portion
    :param portion: portion of the data to use
    :return:
    """
    # get data
    from sklearn.datasets import fetch_20newsgroups
    data_train = fetch_20newsgroups(categories=categories, subset='train', remove=('headers', 'footers', 'quotes'),
                                    random_state=21)
    data_test = fetch_20newsgroups(categories=categories, subset='test', remove=('headers', 'footers', 'quotes'),
                                   random_state=21)

    # train
    train_len = int(portion*len(data_train.data))
    x_train = np.array(data_train.data[:train_len])
    y_train = data_train.target[:train_len]
    # remove empty entries
    non_empty = x_train != ""
    x_train, y_train = x_train[non_empty].tolist(), y_train[non_empty].tolist()

    # test
    x_test = np.array(data_test.data)
    y_test = data_test.target
    non_empty = np.array(x_test) != ""
    x_test, y_test = x_test[non_empty].tolist(), y_test[non_empty].tolist()
    return x_train, y_train, x_test, y_test

def generate_datasets(portion, batch_size=32):
    """
    """
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)
    vectorizer = TfidfVectorizer(max_features=2000)

    x_train = vectorizer.fit_transform(x_train).toarray()
    trainset = torch.utils.data.TensorDataset(
        torch.tensor(x_train).float(), torch.tensor(y_train).long())
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size)
    
    x_test = vectorizer.transform(x_test).toarray()
    testset = torch.utils.data.TensorDataset(
        torch.tensor(x_test).float(), torch.tensor(y_test).long())
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size)

    return trainloader, testloader

def train_model(model, trainloader, epochs=10, lr=0.01):
    """
    Training the model on the training set.
    :param model: The model to train.
    :param trainloader: The training set.
    :param test_loader: The testing set.
    :param epochs: The number of epochs to train the model.
    :param lr: The learning rate for the optimizer.
    :return: The losses and accuracies of the model during training, for each epoch.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    losses = []
    accuracies = []

    for ep in range(epochs):

        print(f'[TRAIN] Epoch: {ep + 1}')
        model.train()
        pred_correct = 0
        ep_loss = 0.

        for i, (inputs, labels) in enumerate(tqdm(trainloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            
            pred_correct += (outputs.argmax(dim=1) == labels).sum().item()
            ep_loss += loss.item()

        accuracies.append(pred_correct / len(trainloader.dataset))
        losses.append(ep_loss / len(trainloader))

    print('[TRAIN] Final Loss: ', losses[-1])
    print('[TRAIN] Final Accuracy: ', accuracies[-1])

    return losses, accuracies

# Q1,2
def MLP_classification(portion=1., model=None):
    """
    Perform linear classification
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    trainloader, testloader = generate_datasets(portion)
    model = nn.Linear(2000, 4)
    losses, accuracies = train_model(model, trainloader, epochs=10, lr=0.1)
    
    plt.plot(accuracies, label='Train')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy')
    plt.show()
    plt.close()

    plt.plot(losses, label='Train')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.show()
    plt.close()

# Q3
def transformer_classification(portion=1.):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from torch.utils.data import DataLoader
    import evaluate
    from tqdm import tqdm

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

    def train_epoch(model, data_loader, optimizer, dev='cpu'):
        """
        Perform an epoch of training of the model with the optimizer
        :param model:
        :param data_loader:
        :param optimizer:
        :param dev:
        :return: Average loss over the epoch
        """
        model.train()
        total_loss = 0.
        # iterate over batches
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(dev)
            attention_mask = batch['attention_mask'].to(dev)
            labels = batch['labels'].to(dev)
            ########### add your code here ###########
        return

    def evaluate_model(model, data_loader, dev='cpu', metric=None):
        model.eval()
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(dev)
            attention_mask = batch['attention_mask'].to(dev)
            labels = batch['labels'].to(dev)
            ########### add your code here ###########
        return

    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    # Parameters
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_labels = len(category_dict)
    epochs = 3
    batch_size = 16
    learning_rate = 5e-5

    # Model, tokenizer, and metric
    model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base', num_labels=num_labels).to(dev)
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    metric = evaluate.load("accuracy")

    # Datasets and DataLoaders
    train_dataset = Dataset(tokenizer(x_train, truncation=True, padding=True), y_train)
    val_dataset = Dataset(tokenizer(x_test, truncation=True, padding=True), y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    ########### add your code here ###########
    return


if __name__ == "__main__":
    portions = [0.1, 0.2, 0.5, 1.]
    # Q1 - single layer MLP
    MLP_classification()

    # Q2 - multi-layer MLP
    pass

    # Q3 - Transformer
    # print("\nTransformer results:")
    # for p in portions[:2]:
    #     print(f"Portion: {p}")
    #     transformer_classification(portion=p)
