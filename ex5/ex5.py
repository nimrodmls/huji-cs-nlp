import wikipedia
import spacy

def get_relation_tokens(tokens):
    """
    Checking whether the given sequence of tokens contain any verbs,
    and that the sequence doesn't contain any punctuation.
    """
    relation_tokens = []
    existing_verbs = False
    for token in tokens:
        if token.pos_ == "VERB":
            relation_tokens.append(token)
            existing_verbs = True

        elif token.pos_ == "ADP":
            relation_tokens.append(token)

        elif token.pos_ == "PUNCT":
            return []
        
    return relation_tokens if existing_verbs else []

def pos_extractor(doc):
    """
    An extractor based on POS tags of tokens in the given document.
    """

    # Extract sequences of proper nouns [(start index, end index),]
    propn_seqs = []
    seq_start_idx = -1
    for idx, token in enumerate(doc):
        # Save the start index of a sequence of proper nouns
        if token.pos_ == "PROPN" and seq_start_idx == -1:
            seq_start_idx = idx
        # Save the end index of a sequence of proper nouns
        elif token.pos_ != "PROPN" and seq_start_idx != -1:
            propn_seqs.append((seq_start_idx, idx-1))
            seq_start_idx = -1
    

    triplets = []
    # Finding all consecutive pairs of proper nouns which do not contain any punctuation
    # and have at least one verb in between (this code can be optimized to run within
    # the extraction of proper noun sequences, above)
    for (seq1_start, seq1_end), (seq2_start, seq2_end) in zip(propn_seqs, propn_seqs[1:]):
        tokens = doc[seq1_end+1:seq2_start] # All relevant tokens
        relation_tokens = get_relation_tokens(tokens)
        if relation_tokens:
            triplet = (doc[seq1_start:seq1_end+1], relation_tokens, doc[seq2_start:seq2_end+1])
            print(triplet)
            triplets.append(triplet)
            
    return triplets

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    wikipedia.set_lang("en")
    page = wikipedia.page('Natural Language Processing')
    doc = nlp(page.content)

    pos_extractor(doc)