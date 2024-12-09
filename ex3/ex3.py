import nltk
from sklearn.model_selection import train_test_split

class MostLikelyTag():
    """
    A class that assigns the most likely tag to a word, based on the training data,
    using MLE for the estimation of the probabilities.
    """
    def __init__(self):
        self.tag_count = {} # Count of each tag {tag: count}
        self.word_tag_count = {} # Count of each word for each tag {word: {tag: count}}

    def train(self, train_data):
        """
        Trains the model with the training data
        """
        # Each sentence is split to words and their respective tags (as tuples)
        for sentence in train_data:
            for word, tag in sentence:

                # Counting the tags                
                if tag not in self.tag_count:
                    self.tag_count[tag] = 0
                self.tag_count[tag] += 1

                # Counting the words for each tag
                if word not in self.word_tag_count:
                    self.word_tag_count[word] = {}
                if tag not in self.word_tag_count[word]:
                    self.word_tag_count[word][tag] = 0
                self.word_tag_count[word][tag] += 1

    def predict(self, word):
        # Handling unknown words (e.g. words that are not in the training data)
        # by assigning the tag 'NN' as defined in the assignment
        if word not in self.word_tag_count:
            return 'NN'
        
        # Finding the tag with the highest probability for the word
        max_prob = 0
        max_tag = ''
        for tag in self.word_tag_count[word]:
            prob = self.word_tag_count[word][tag] / self.tag_count[tag]
            if prob > max_prob:
                max_prob = prob
                max_tag = tag

        return max_tag
    
class Bigram_HMM():
    
    def __init__(self):
        self.transition_probs = {} # Transition probabilities {tag1: {tag2: prob}}
        self.emission_probs = {} # Emission probabilities {tag: {word: prob}}

    def train(self, train_data):
        # Computing the transition & emission probabilities using MLE

        tag_transitions = {} # {tag1: {tag2: count}} - Count of each tag transition (bigram)
        tag_word_count = {} # Count of each word for each tag {tag: {word: count}}

        for sentence in train_data:
            for word, tag in sentence:

                # Counting the transitions between tags - for transition probabilities
                if tag not in tag_transitions:
                    tag_transitions[tag] = {}
                if tag not in tag_transitions[tag]:
                    tag_transitions[tag][tag] = 0
                tag_transitions[tag][tag] += 1

                # Counting the words for each tag - for emission probabilities
                if tag not in tag_word_count:
                    tag_word_count[tag] = {}
                if word not in tag_word_count[tag]:
                    tag_word_count[tag][word] = 0
                tag_word_count[tag][word] += 1

        self.transition_probs = tag_transitions
        self.emission_probs = tag_word_count

        for tag1 in self.transition_probs:

            # Computing the transition probabilities
            for tag2 in self.transition_probs[tag1]:
                transition_count = sum(self.transition_probs[tag1].values())
                # Calculating the probability by normalizing with the total count of transitions from tag1
                self.transition_probs[tag1][tag2] = tag_transitions[tag1][tag2] / transition_count

            # Computing the emission probabilities
            for word in self.emission_probs[tag1]:
                word_count = sum(self.emission_probs[tag].values())
                # Calculating the probability by normalizing with the total count of words for the tag
                self.emission_probs[tag1][word] = tag_word_count[tag1][word] / word_count

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
    # Task A - Getting the dataset
    train_set, test_set = get_dataset()
    
    # Task B - Training a Most Likely Tag baseline
    mlt = MostLikelyTag()
    mlt.train(train_set)

    ### TODO: Add evaluation here
    

if __name__ == '__main__':
    main()