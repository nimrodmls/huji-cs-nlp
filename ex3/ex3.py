import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

START_TOK = '<START>'
START_TAG = START_TOK
STOP_TOK = '<STOP>'
STOP_TAG = STOP_TOK

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
    
class Bigram_HMM_Tagger():
    """
    Text tagging model using a Bigram Hidden Markov Model
    NOTE: Implemented all probabilities in log space to avoid diminishing probabilities
    """
    PSEUDOWORDS_THRESHOLD = 5 # Frequency threshold for pseudowords
    PSEDUOWORDS_SUFFIXES = {
        '\'s': 'Association', 's\'': 'Association', # Associative suffixes (e.g. John's)
        'teen': 'NumberWord', # A number represented in its word form (e.g. seventeen)
    }
    
    def __init__(self, is_pseudowords=False):
        """
        :param pseudowords: Whether to use pseudowords or not (in training & prediction)
        """
        self.is_pseudowords = is_pseudowords

        # Transition probabilities {tag1: {tag2: prob}}
        self.transition_probs = {} 

        # Emission probabilities {tag: {word: prob}}
        self.emission_probs = {} 

        self.vocab = set() # The vocabulary of the training data

    def _get_pseudoword(self, word):
        """
        Given a word, the function returns its pseudoword representation
        This function DOES NOT check for the frequency of the given word
        """
        # Checking for suffixes
        for suffix in self.PSEDUOWORDS_SUFFIXES:
            # A word is considered suffixid if it ends with the suffix and is longer than the suffix
            if word.endswith(suffix) and len(word) > len(suffix):
                return self.PSEDUOWORDS_SUFFIXES[suffix]

        # Checking for monetary values (e.g. 1,000.0)
        is_monetary = True
        is_symboled = False
        for idx, ch in enumerate(word):
            # The first character can be a digit or a currency symbol
            if idx == 0 and ch in ['$', '€', '£']:
                is_symboled = True
                continue
            # The currency symbol should be followed by a digit
            elif idx == 1 and is_symboled and not ch.isdigit():
                is_monetary = False
                break 
            # The rest of the characters should be digits or a comma or a dot
            elif not (ch.isdigit() or ch in [',', '.']):
                is_monetary = False
                break
        
        if is_monetary:
            return 'MonetaryValue'
        
        # Different number cases (e.g. 1997, 97)
        if word.isdigit():
            if len(word) == 4:
                return 'FourDigitNumber'
            elif len(word) == 3:
                return 'ThreeDigitNumber'
            elif len(word) == 2:
                return 'TwoDigitNumber'
                
        # Checking for all caps words (e.g. JOHN)
        if word.isupper():
            return 'AllCaps'
        
        # Checking for all lowercase (e.g. john)
        if word.islower():
            return 'AllLower'

        # Checking for initial caps (e.g. John)
        if word[0].isupper():
            return 'InitialCaps'
        
        # Hyphen-separated words (e.g. part-time)
        if len(word.split('-')) > 1:
            return 'HyphenatedWord'

    def train(self, train_data, add_one_smoothing=False):

        # {tag1: {tag2: count}} - Count of each tag transition (bigram)
        tag_transitions = {}

         # Count of each word for each tag {tag: {word: count}}
        tag_word_count = {START_TAG: {START_TOK: 0}, STOP_TAG: {STOP_TOK: 0}}

        # Counting and then computing the transition & emission probabilities using MLE
        for sentence in train_data:
            # Iterating on all pairs of words and their tags in the sentence
            for i in range(1, len(sentence)):
                word, tag = sentence[i]
                prev_word, prev_tag = sentence[i-1]

                # Adding the word to the vocabulary
                self.vocab.add(word)

                # Counting the transitions between tags - for transition probabilities
                if prev_tag not in tag_transitions:
                    tag_transitions[prev_tag] = {}
                if tag not in tag_transitions[prev_tag]:
                    tag_transitions[prev_tag][tag] = 0
                tag_transitions[prev_tag][tag] += 1

                # Counting the words for each tag - for emission probabilities
                if prev_tag not in tag_word_count:
                    tag_word_count[prev_tag] = {}
                if prev_word not in tag_word_count[prev_tag]:
                    tag_word_count[prev_tag][prev_word] = 0
                tag_word_count[prev_tag][prev_word] += 1
    
        tag_word_count[STOP_TAG][STOP_TOK] = tag_word_count[START_TAG][START_TOK]

        self.transition_probs = tag_transitions
        self.emission_probs = tag_word_count

        for tag1 in self.transition_probs:
            # Computing the transition probabilities
            transition_count = sum(self.transition_probs[tag1].values())
            for tag2 in self.transition_probs[tag1]:
                # MLE
                self.transition_probs[tag1][tag2] = np.log(self.transition_probs[tag1][tag2] / transition_count)

            # Computing the emission probabilities
            word_count = sum(self.emission_probs[tag].values())
            for word in self.emission_probs[tag1]:
                if add_one_smoothing:
                    # Add-one smoothing
                    self.emission_probs[tag1][word] = np.log((self.emission_probs[tag1][word] + 1) / (word_count + len(self.vocab)))
                else:
                    # Purely MLE
                    self.emission_probs[tag1][word] = np.log(self.emission_probs[tag1][word] / word_count)
    
    def predict(self, sentence):
        """
        Runs the Viterbi algorithm to find the most likely tag sequence for the sentence
        Note that the algorithm runs at each call, as it's sensitive to the length of the given sentence
        """
        # The viterbi table - {t: {tag: prob}}, the list is ordered by the observations
        # The table is initialized for the first observation (assumed to be start of sentence)
        viterbi = {0: {tag: -np.inf for tag in self.transition_probs}}
        viterbi[0][START_TAG] = 0
        backpointer = {0: {START_TAG: None}}
        T = len(sentence) # Length of the sentence (number of observations)

        # Iterating through all the observations, except the start
        for t in range(1, T):
            viterbi[t] = {}
            backpointer[t] = {}

            for tag in self.transition_probs:
                max_prob = -np.inf
                max_tag = None
                for prev_tag in self.transition_probs:
                    # Computing the probability of the transition
                    transition_prob = -np.inf
                    if tag in self.transition_probs[prev_tag]:
                        transition_prob = self.transition_probs[prev_tag][tag]
                    # Computing the probability of the emission
                    emission_prob = -np.inf
                    if sentence[t] in self.emission_probs[tag]:
                        emission_prob = self.emission_probs[tag][sentence[t]]
                    # Computing the probability of the current tag sequence
                    prob = viterbi[t-1][prev_tag] + transition_prob + emission_prob
                    if prob > max_prob:
                        max_prob = prob
                        max_tag = prev_tag
                # Updating the table according to the max probability of the current tag
                viterbi[t][tag] = max_prob
                backpointer[t][tag] = max_tag
            
            if max(viterbi[t].values()) == -np.inf:
                # Handling unknown words by assigning the tag 'NN' (arbitrary choice)
                viterbi[t]['NN'] = 0
                backpointer[t]['NN'] = list(viterbi[t-1].keys())[np.argmax(list(viterbi[t-1].values()))]

        # Finding the most likely tag sequence by backtracking
        max_prob = -np.inf
        max_tag = None
        for tag in viterbi[T-1]:
            prob = viterbi[T-1][tag]
            if prob > max_prob:
                max_prob = prob
                max_tag = tag

        # Building the tag sequence by backtracking
        tag_sequence = [max_tag]
        for t in range(T-2, -1, -1):
            max_tag = backpointer[t + 1][max_tag]
            tag_sequence.insert(0, max_tag)

        return tag_sequence

def get_dataset():
    """
    Retreives the tagged sentences of the 'news' category of the brown corpus.
    The sentences are split to 90% training and 10% testing.
    """
    all_data = nltk.corpus.brown.tagged_sents(categories='news')
    # Artifically adding the START/END tokens & tags to the sentences
    all_data = [[(START_TOK, START_TAG)] + sentence + [(STOP_TOK, STOP_TAG)] for sentence in all_data]
    train_data, test_data = train_test_split(all_data, test_size=0.1, shuffle=False) # TODO: Change shuffle to true
    return train_data, test_data

def download_corpus():
    """
    Downloads the brown corpus, if not already downloaded
    """
    nltk.download('brown')

def bigram_hmm_experiment(train_set, test_set, add_one_smoothing=False):
    """
    """
    bigram_hmm = Bigram_HMM_Tagger()
    bigram_hmm.train(train_set, add_one_smoothing=add_one_smoothing)
    test_x = []
    test_y = []
    for sentence in test_set:
        sent_words = []
        sent_tags = []

        for word, tag in sentence:
            sent_words.append(word)
            sent_tags.append(tag)

        test_x.append(sent_words)
        test_y.append(sent_tags)

    correct_preds = 0
    total_preds = 0
    for x, y in tqdm(list(zip(test_x, test_y))):
        y_pred = bigram_hmm.predict(x)

        total_preds += len(y)
        # Counting the correct predictions, while ignoring the STOP tokens (last word)
        for true, pred in zip(y[:-1], y_pred[:-1]):
            if true == pred:
                correct_preds += 1

    accuracy = correct_preds / total_preds
    print(f'Error rate: {1 - accuracy}')

def task_3(train_set, test_set):
    """
    Runs the experiments for Task 3
    """
    bigram_hmm_experiment(train_set, test_set, add_one_smoothing=False)

def task_4(train_set, test_set):
    """
    Runs the experiments for Task 4
    """
    bigram_hmm_experiment(train_set, test_set, add_one_smoothing=True)

def corpus_visualize_frequency_distribution(corpus):
    """
    """
    mlt = MostLikelyTag()
    mlt.train(corpus)

    freqs = []
    for word in mlt.word_tag_count:
        freqs.append(sum(mlt.word_tag_count[word].values()))

    bins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plt.hist(freqs, bins=bins)
    plt.xticks(bins)
    plt.xlabel('Frequency')
    plt.ylabel('Number of words')
    plt.title('Frequency Distribution of Words in the Corpus')
    plt.savefig("frequency_distribution.png")

def corpus_get_unique_words(corpus, threshold=5):
    """
    :param corpus: List of sentences
    :param threshold: Frequency threshold for uniqueness
    """
    mlt = MostLikelyTag()
    mlt.train(corpus)
    unique_words = [word for word in mlt.word_tag_count if sum(mlt.word_tag_count[word].values()) <= threshold]
    return unique_words

def main():
    ### Task A - Getting the dataset
    download_corpus()
    train_set, test_set = get_dataset()
    
    ### Task B - Training a Most Likely Tag baseline
    # mlt = MostLikelyTag()
    # mlt.train(train_set)

    ### TODO: Add evaluation here

    ### Task C - Bigram HMM Tagger
    # task_3(train_set, test_set)

    ### Task D - Bigram HMM Tagger with Add-One Smoothing
    # task_4(train_set, test_set)

    ### Task C - Bigram HMM Tagger with Pseudowords
    # Doing some analysis first, for choosing the pseudowords
    corpus_visualize_frequency_distribution(train_set)
    unique_words = corpus_get_unique_words(
        train_set, threshold=3)
    with open("unique_words.txt", "w") as f:
        for word in unique_words:
            f.write(f'{word}\n')
    

if __name__ == '__main__':
    main()