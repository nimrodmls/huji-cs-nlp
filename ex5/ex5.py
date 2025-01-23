from typing import List
import wikipedia
import spacy
import itertools
import random
import google.generativeai as genai

TRUMP_GEMINI_QUERY = """
Using the text below, extract all of the triplets of text which correspond to the template of (Subject, Relation, Object) where the Subject and Object slot fillers are names (proper nouns), and the Relation slot filler is a verb or a verb along with a preposition
Starting in 1968, Trump was employed at his father's real estate company, Trump Management, which owned racially segregated middle-class rental housing in New York City's outer boroughs.[9][10] In 1971, his father made him president of the company and he began using the Trump Organization as an umbrella brand.[11] Roy Cohn was Trump's fixer, lawyer, and mentor for 13 years in the 1970s and 1980s.[12] In 1973, Cohn helped Trump countersue the U.S. government for $100 million (equivalent to $686 million in 2023)[13] over its charges that Trump's properties had racial discriminatory practices. Trump's counterclaims were dismissed, and the government's case was settled with the Trumps signing a consent decree agreeing to desegregate.[14] Helping Trump projects, Cohn was a consigliere whose Mafia connections controlled construction unions.[15] Cohn introduced political consultant Roger Stone to Trump, who enlisted Stone's services to deal with the federal government.[16] Between 1991 and 2009, he filed for Chapter 11 bankruptcy protection for six of his businesses: the Plaza Hotel in Manhattan, the casinos in Atlantic City, New Jersey, and the Trump Hotels & Casino Resorts company.[17]
"""

def gemini_query(prompt: str):
    """
    Makes a query to Gemini LLM model with the given prompt,
    and returns the response
    """
    # NOTE: The API key has been omitted for security reasons
    genai.configure(api_key="")
    model = genai.GenerativeModel("gemini-1.5-flash")
    return model.generate_content(prompt).parts[0].text

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
            triplet = (h1_compounds, [h1.head], h2_compounds)
            triplets.append(triplet)
        # The second condition as defined in the exercise
        elif (h1.head == h2.head.head) and (h1.dep_ == "nsubj") and (h2.dep_ == "pobj") and (h2.head.dep_ == "prep"):
            h1_compounds = get_corresponding_compound(h1)
            h2_compounds = get_corresponding_compound(h2)
            triplet = (h1_compounds, [h1.head, h2.head], h2_compounds)
            triplets.append(triplet)

    return triplets

def evaluate_extractors(doc):
    """
    Evaluates the two extractors on the given document.
    """
    pos_triplets = pos_extractor(doc)
    dep_triplets = dependency_tree_extractor(doc)

    print(f'POS extractor triplets: {len(pos_triplets)}')
    print(f'Dependency Tree extractor triplets: {len(dep_triplets)}')

    random_sample_cnt = 5
    # Take random sample of triplets from each extractor
    print(f'\nPOS extractor random sample:')
    for triplet in random.sample(pos_triplets, random_sample_cnt):
        print(triplet)

    print(f'\nDependency Tree extractor random sample:')
    for triplet in random.sample(dep_triplets, random_sample_cnt):
        print(triplet)

def wikipedia_pages_evaluation(pages: List[str]):
    """
    """
    nlp = spacy.load("en_core_web_sm")
    wikipedia.set_lang("en")
    for page in pages:
        print(f'\n --- Evaluating page: {page} ---')
        doc = nlp(wikipedia.page(page).content)
        evaluate_extractors(doc)

def main():
    # pages = ["Donald Trump", "Ruth Bader Ginsburg", "J.K. Rowling"]
    # wikipedia_pages_evaluation(pages)

    response = gemini_query(TRUMP_GEMINI_QUERY)

if __name__ == "__main__":
    main()