import numpy as np
import spacy
from datasets import load_dataset
import pickle
import tqdm

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
    
    token_text_pair = (get_token_text(token_pair[0]), get_token_text(token_pair[1]))
    add_token_to_frequencies(token_text_pair, frequencies)

def process_line_token_frequencies(line, unigram_frequencies, bigram_frequencies):
    """
    """
    last_token = None
    for token_pair in zip([LINE_START] + list(line), list(line[1:]) + [LINE_END]):
        ### Unigram handling
        add_token_to_frequencies(token_pair[0], unigram_frequencies)

        ### Bigram handling
        if not is_token_valid(token_pair[1]) and is_token_valid(token_pair[0]):
            last_token = token_pair[0]

        elif not is_token_valid(token_pair[0]) and is_token_valid(token_pair[1]) and last_token is not None:
            add_token_pair_to_frequencies((last_token, token_pair[1]), bigram_frequencies)
            last_token = None

        elif is_token_valid(token_pair[0]) and is_token_valid(token_pair[1]):
            add_token_pair_to_frequencies(token_pair, bigram_frequencies)

    ### Unigram - Artificially adding the end token
    add_token_to_frequencies(LINE_END, unigram_frequencies)

def main():
    nlp = spacy.load("en_core_web_sm")
    text = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    
    print("Processing token frequencies...")

    unigram_frequencies = {}
    bigram_frequencies = {}
    for row in tqdm.tqdm(text['text']):
        process_line_token_frequencies(nlp(row), unigram_frequencies, bigram_frequencies)

    # For testing, to save time
    pickle.dump(unigram_frequencies, open("unigram_frequencies_dict.p", "wb"))
    pickle.dump(bigram_frequencies, open("bigram_frequencies_dict.p", "wb"))
    
if __name__ == "__main__":
    main()  