import numpy as np
import spacy
from datasets import load_dataset
import pickle
import tqdm
import random
import math

LINE_START = "[START]"
LINE_END = "[END]"

def is_token_valid(token):
    """
    Validating whether the token is one we'd like to consider, conditions:
    - Token is not a punctuation mark
    """
    return (token in [LINE_START, LINE_END]) or token.is_alpha

def get_token_text(token):
    """
    """
    if token in [LINE_START, LINE_END]:
        return token
    return token.lemma_

def add_token_to_frequencies(token, frequencies):
    if token not in frequencies:
        frequencies[token] = 1
    else:
        frequencies[token] += 1

def add_token_pair_to_frequencies(token_pair, frequencies):
    """
    """
    # Safety assert - This shouldn't happen
    if not is_token_valid(token_pair[0]) or not is_token_valid(token_pair[1]):
        raise ValueError("Both tokens should be valid, got: ", token_pair)
    
    first_token = get_token_text(token_pair[0])
    second_token = get_token_text(token_pair[1])

    if first_token not in frequencies:
        frequencies[first_token] = {second_token: 1}
    else:
        if second_token not in frequencies[first_token]:
            frequencies[first_token][second_token] = 1
        else:
            frequencies[first_token][second_token] += 1

def process_line_token_frequencies(line, unigram_frequencies, bigram_frequencies):
    """
    """
    last_token = None
    for token_pair in zip([LINE_START] + list(line), list(line) + [LINE_END]):
        ### Unigram handling
        add_token_to_frequencies(get_token_text(token_pair[0]), unigram_frequencies)

        ### Bigram handling

        # If the next token is not valid, we will skip it and keep the last token
        # until we find a valid token to pair it with
        if not is_token_valid(token_pair[1]) and is_token_valid(token_pair[0]):
            last_token = token_pair[0]

        elif not is_token_valid(token_pair[0]) and is_token_valid(token_pair[1]) and last_token is not None:
            add_token_pair_to_frequencies((last_token, token_pair[1]), bigram_frequencies)
            last_token = None

        elif is_token_valid(token_pair[0]) and is_token_valid(token_pair[1]):
            add_token_pair_to_frequencies(token_pair, bigram_frequencies)

    ### Unigram - Artificially adding the end token
    add_token_to_frequencies(LINE_END, unigram_frequencies)

def train_unigram_bigram_models(nlp):
    """
    """
    text = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

    # The unigram frequencies will be a dictionary with the token as the 
    # key and the frequency as the value
    unigram_frequencies = {}
    # The bigram frequencies will be a dictionary with the first token as the key 
    # and the value will be another dictionary with the second token as the key 
    # and the frequency of the pair (first followed by second) as the value
    bigram_frequencies = {}
    for row in tqdm.tqdm(text['text']):
        process_line_token_frequencies(nlp(row), unigram_frequencies, bigram_frequencies)

    # For testing, to save time
    pickle.dump(unigram_frequencies, open("unigram_frequencies_dict.p", "wb"))
    pickle.dump(bigram_frequencies, open("bigram_frequencies_dict.p", "wb"))

    return unigram_frequencies, bigram_frequencies

def load_pretrained_models():
    """
    """
    unigram_frequencies = pickle.load(open("unigram_frequencies_dict.p", "rb"))
    bigram_frequencies = pickle.load(open("bigram_frequencies_dict.p", "rb"))

    return unigram_frequencies, bigram_frequencies

def bigram_predict_next_token(first_token, bigram_frequencies):
    """
    """
    total_occurances = sum(bigram_frequencies[first_token].values())
    next_token_chance = random.randint(0, total_occurances)
    for token, frequency in bigram_frequencies[first_token].items():
        next_token_chance -= frequency
        if next_token_chance <= 0:
            return token
        
def bigram_get_next_token_probabilities(first_token, bigram_frequencies):
    """
    """
    total_occurances = sum(bigram_frequencies[first_token].values())
    probabilities = {}
    for token, frequency in bigram_frequencies[first_token].items():
        probabilities[token] = math.log(frequency / total_occurances)

    return probabilities

def bigram_get_sentence_probability(sentence_tokens, bigram_frequencies):
    """
    """
    prob_log_sum = 0
    for token in zip([LINE_START] + list(sentence_tokens), list(sentence_tokens) + [LINE_END]):
        probs = bigram_get_next_token_probabilities(get_token_text(token[0]), bigram_frequencies)
        token2_text = get_token_text(token[1])
        if token2_text in probs:
            prob_log_sum += probs[token2_text]
        else:
            return 0
    return prob_log_sum

def compute_perplexity(sentences_tokens, bigram_frequencies):
    total_log_prob = 0
    total_tokens = 0
    for sentence_tokens in sentences_tokens:
        prob = bigram_get_sentence_probability(sentence_tokens, bigram_frequencies)
        total_log_prob += prob
        total_tokens += len(sentence_tokens) + 2  # +2 for the START and sEND token
    return math.exp(-total_log_prob / total_tokens)

def main():    
    ### Loading / Training the models (Q1)
    nlp = spacy.load("en_core_web_sm")
    load_pretrained = True

    print("Processing token frequencies...")
    unigram_freqs, bigram_freqs = load_pretrained_models() if load_pretrained else train_unigram_bigram_models(nlp)

    ### Testing the models

    ### Q2
    probs = bigram_get_next_token_probabilities('in', bigram_freqs)
    print(f"I have a house in {max(probs, key=probs.get)}")

    ### Q3
    sentence_1 = "Brad Pitt was born in Oklahoma"
    sentence_2 = "The actor was born in USA"

    ## Q3.1
    sentence_1_tokens = nlp(sentence_1)
    sentence_2_tokens = nlp(sentence_2)
    print(f"Probability of '{sentence_1}': {bigram_get_sentence_probability(sentence_1_tokens, bigram_freqs)}")
    print(f"Probability of '{sentence_2}': {bigram_get_sentence_probability(sentence_2_tokens, bigram_freqs)}")

    ## Q3.2
    sentences_tokens = [sentence_1_tokens, sentence_2_tokens]
    perplexity = compute_perplexity(sentences_tokens, bigram_freqs)
    print(f"Perplexity of both sentences: {perplexity}")

    ### Q4



if __name__ == "__main__":
    main()  