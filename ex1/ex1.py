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
    return token.is_alpha

def process_line_token_frequencies(line, frequencies):
    """
    """
    # Each line has a start and an end, we add it artificially
    frequencies[LINE_START] += 1
    frequencies[LINE_END] += 1

    # Processing the line
    #for token_pairs in zip(line, line[1:]):
    for token in line:
        if is_token_valid(token):
            token_text = token.lemma_
            if token_text not in frequencies:
                frequencies[token_text] = 1
            else:
                frequencies[token_text] += 1

def main():
    nlp = spacy.load("en_core_web_sm")
    text = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    
    print("Processing token frequencies...")

    frequencies = {LINE_START: 0, LINE_END: 0}
    for row in tqdm.tqdm(text['text']):
        process_line_token_frequencies(nlp(row), frequencies)

    # For testing, to save time
    pickle.dump(frequencies, open("frequencies_dict.p", "wb"))
    


if __name__ == "__main__":
    main()  