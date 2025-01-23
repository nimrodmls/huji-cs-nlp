import wikipedia
import spacy
import itertools

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

def get_corresponding_compound(token):
    """
    Returns the corresponding compound token of the given token, with the token itself
    """
    return [child for child in token.children if child.dep_ == "compound"] + [token]

def dependency_tree_extractor(doc):
    """
    An extractor based on the dependency tree of the given document.
    """

    propn_heads = []
    for token in doc:
        # Checking if the current token is a proper noun head
        if token.pos_ == "PROPN" and token.dep_ != "compound":
            propn_heads.append(token)
    
    triplets = []
    for h1, h2 in itertools.combinations(propn_heads, 2):
        # The first condition as defined in the exercise
        if (h1.head == h2.head) and (h1.dep_ == "nsubj") and (h2.dep_ == "dobj"):
            h1_compounds = get_corresponding_compound(h1)
            h2_compounds = get_corresponding_compound(h2)
            triplet = (h1_compounds, h1.head, h2_compounds)
            triplets.append(triplet)
        # The second condition as defined in the exercise
        elif (h1.head == h2.head.head) and (h1.dep_ == "nsubj") and (h2.dep_ == "pobj") and (h2.head.dep_ == "prep"):
            h1_compounds = get_corresponding_compound(h1)
            h2_compounds = get_corresponding_compound(h2.head)
            triplet = (h1_compounds, h1.head, h2_compounds)
            triplets.append(triplet)


    #return triplets

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    wikipedia.set_lang("en")
    page = wikipedia.page('Natural Language Processing')
    doc = nlp(page.content)

    # pos_extractor(doc)
    dependency_tree_extractor(doc)