import numpy as np
import spacy
from datasets import load_dataset
import pickle
import tqdm
import math

LINE_START = "[START]"


def is_token_valid(token):
    """
    Validating whether the token is one we'd like to consider, conditions:
    - Token is not a punctuation mark
    """
    return (token == LINE_START) or token.is_alpha


def get_token_text(token):
    """
    """
    if token == LINE_START:
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
    line = [token for token in line if is_token_valid(token)]
    if len(line) == 0:
        return

    for token_pair in zip([LINE_START] + list(line), list(line)):

        ### Unigram handling
        add_token_to_frequencies(get_token_text(token_pair[0]), unigram_frequencies)

        add_token_pair_to_frequencies(token_pair, bigram_frequencies)

    # Adding last token to unigram
    add_token_to_frequencies(get_token_text(line[-1]), unigram_frequencies)
    

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


def bigram_get_next_token_probabilities(first_token, bigram_frequencies):
    """
    """
    total_occurances = sum(bigram_frequencies[first_token].values())
    probabilities = {}
    for token, frequency in bigram_frequencies[first_token].items():
        probabilities[token] = np.log(frequency / total_occurances)

    return probabilities


def bigram_get_sentence_probability(sentence_tokens, bigram_frequencies):
    """
    """
    prob_log_sum = 0
    for token in zip([LINE_START] + list(sentence_tokens), list(sentence_tokens)):
        probs = bigram_get_next_token_probabilities(get_token_text(token[0]), bigram_frequencies)
        token2_text = get_token_text(token[1])
        if token2_text in probs:
            prob_log_sum += probs[token2_text]
        else:
            return -np.inf  # Return -inf for zero probability instead of 0
    return prob_log_sum



def compute_perplexity(sentences_tokens, bigram_frequencies):
    total_log_prob = 0
    total_tokens = 0
    for sentence_tokens in sentences_tokens:
        prob = bigram_get_sentence_probability(sentence_tokens, bigram_frequencies)
        total_log_prob += prob
        total_tokens += len(sentence_tokens)
    return math.exp(-total_log_prob / total_tokens)


def combined_get_sentence_probability(sentence_tokens, unigram_frequencies, bigram_frequencies, lambda_bigram, lambda_unigram):
    """
    """
    total_log_prob = 0
    total_unigrams = sum(unigram_frequencies.values())

    for token_pair in zip([LINE_START] + list(sentence_tokens), list(sentence_tokens)):
        first_token = get_token_text(token_pair[0])
        second_token = get_token_text(token_pair[1])

        # Unigram probability
        unigram_count = unigram_frequencies.get(second_token, 0)
        unigram_prob = unigram_count / total_unigrams

        #Bigram probability
        if first_token in bigram_frequencies and second_token in bigram_frequencies[first_token]:
            bigram_count = bigram_frequencies[first_token][second_token]
            total_bigram_count = sum(bigram_frequencies[first_token].values())
            bigram_prob = bigram_count / total_bigram_count
        else:
            bigram_prob = 0

        # Combined probability using the weighted sum of unigram and bigram probabilities
        combined_prob = lambda_bigram * bigram_prob + lambda_unigram * unigram_prob

        # Apply log if combined probability is greater than 0, else use -inf for zero probability
        if combined_prob > 0:
            total_log_prob += math.log(combined_prob)
        else:
            return -np.inf

    return total_log_prob


def compute_combined_perplexity(sentences_tokens, unigram_frequencies, bigram_frequencies, lambda_bigram=2 / 3,
                                lambda_unigram=1 / 3):
    total_log_prob = 0
    total_tokens = 0

    for sentence_tokens in sentences_tokens:
        prob_log_sum = combined_get_sentence_probability(
            sentence_tokens,
            unigram_frequencies,
            bigram_frequencies,
            lambda_bigram,
            lambda_unigram
        )
        if prob_log_sum == -np.inf:
            return float('inf')

        total_log_prob += prob_log_sum
        total_tokens += len(sentence_tokens)

    return math.exp(-total_log_prob / total_tokens)


def main():
    ### Loading / Training the models (Q1)
    nlp = spacy.load("en_core_web_sm")
    load_pretrained = False # On True, a pretrained model will be read, otherwise, train & save

    print("Processing token frequencies...")
    unigram_freqs, bigram_freqs = load_pretrained_models() if load_pretrained else train_unigram_bigram_models(nlp)

    ### Testing the models

    ### Q2
    probs = bigram_get_next_token_probabilities('in', bigram_freqs)
    print("Q2:")
    print(f"I have a house in {max(probs, key=probs.get)}")
    print()

    ### Q3
    sentence_1 = "Brad Pitt was born in Oklahoma"
    sentence_2 = "The actor was born in USA"
    sentence_1_tokens = nlp(sentence_1)
    sentence_2_tokens = nlp(sentence_2)

    ## Q3.1
    print("Q3")
    print(f"Probability of '{sentence_1}': {bigram_get_sentence_probability(sentence_1_tokens, bigram_freqs)}")
    print(f"Probability of '{sentence_2}': {bigram_get_sentence_probability(sentence_2_tokens, bigram_freqs)}")


    ## Q3.2
    sentences_tokens = [sentence_1_tokens, sentence_2_tokens]
    perplexity = compute_perplexity(sentences_tokens, bigram_freqs)
    print(f"Perplexity of both sentences: {perplexity}")
    print()

    ### Q4
    sentences_tokens = [sentence_1_tokens, sentence_2_tokens]
    lambda_bigram = 2 / 3
    lambda_unigram = 1 / 3
    print("Q4")
    print(f"Probability of '{sentence_1}': {combined_get_sentence_probability(sentence_1_tokens,unigram_freqs,bigram_freqs,lambda_bigram, lambda_unigram)}")
    print(f"Probability of '{sentence_2}': {combined_get_sentence_probability(sentence_2_tokens,unigram_freqs,bigram_freqs,lambda_bigram, lambda_unigram)}")
    print(f"Perplexity of both sentences: {compute_combined_perplexity(sentences_tokens,unigram_freqs,bigram_freqs,lambda_bigram, lambda_unigram)}")

if __name__ == "__main__":
    main()
