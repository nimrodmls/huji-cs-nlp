import nltk
from sklearn.model_selection import train_test_split

def get_dataset():
    """
    Retreives the tagged sentences of the 'news' category of the brown corpus.
    The sentences are split to 90% training and 10% testing.
    """
    all_data = nltk.corpus.brown.tagged_sents(categories='news')
    train_data, test_data = train_test_split(all_data, test_size=0.1)
    return train_data, test_data

def download_corpus():
    """
    Downloads the brown corpus, if not already downloaded
    """
    nltk.download('brown')

def main():
    download_corpus()
    dataset = get_dataset()

if __name__ == '__main__':
    main()