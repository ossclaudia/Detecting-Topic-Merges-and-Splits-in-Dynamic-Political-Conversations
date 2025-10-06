# ---------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------Importing Packages----------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

import re
import string
import html
from datetime import timedelta
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
import spacy
from nltk.corpus import stopwords
import nltk
import emoji
from rapidfuzz import process, fuzz
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from sklearn.metrics.pairwise import cosine_similarity
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
nlp_ner = spacy.load("en_core_web_sm")

# ---------------------------------------------------------------------------------------------------------------------------------
# --------------------------------Text Cleaning Utilities--------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

def replace_html(text):
    """Convert HTML entities to readable text."""
    return html.unescape(text) if isinstance(text, str) else ""

def clean_hashtags(hashtags):
    """Remove '#' and replace commas with spaces."""
    if pd.isna(hashtags):
        return ""
    hashtags = re.sub(r"#", "", hashtags)
    hashtags = re.sub(r",", " ", hashtags)
    return hashtags.strip()

def clean_urls(urls):
    """Extract meaningful words from URLs."""
    if pd.isna(urls):
        return ""
    urls = re.sub(r"https?://", "", urls)      # remove http:// or https://
    urls = re.sub(r"www\.", "", urls)          # remove www.
    urls = re.sub(r"[^a-zA-Z/]", " ", urls)    # keep only letters and slashes
    urls = re.sub(r"/", " ", urls)
    urls = re.sub(r"\s+", " ", urls).strip()
    words = urls.split()
    irrelevant = {"com", "org", "gov", "net", "html"}
    words = [w for w in words if w not in irrelevant]
    return " ".join(words)

def clean_tweet_row(row):
    """Clean and normalize a single tweet."""
    text, hashtags, urls = row['text'], row['hashtags'], row['urls']

    # 1. Replace HTML entities (&amp; -> &)
    text = replace_html(text)

    # 2. Clean hashtags and URLs
    hashtags_clean = clean_hashtags(hashtags)
    url_words = clean_urls(urls)

    # 3. Concatenate
    combined = f"{text} {hashtags_clean} {url_words}"

    # 4. Remove RT, @, emojis, punctuation, etc.
    combined = re.sub(r'\bRT\b', '', combined) # RT out 
    combined = emoji.replace_emoji(combined, replace='')
    combined = re.sub(r'http\S+|www\S+', '', combined)
    combined = re.sub(r'#', '', combined)
    combined = re.sub(r'@', '', combined) # Maintain the person's account name, just taking the @ out
    combined = re.sub(r'[^a-zA-Z\s]', ' ', combined)
    combined = combined.lower()
    combined = re.sub(r'\s+', ' ', combined).strip()

    # 5. Remove stopwords
    tokens = [word for word in combined.split() if word not in stop_words]

    # 6. Lemmatization
    doc = nlp(" ".join(tokens))
    lemmas = [token.lemma_ for token in doc]

    return " ".join(lemmas)

# ---------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------Parallel Processing----------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

def parallel_apply(df, func):
    """Apply a function in parallel over dataframe rows."""
    with Pool(cpu_count()) as p:
        result = p.map(func, [row for _, row in df.iterrows()])
    return result

# ---------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------Entity Normalization---------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

def normalize_entities(text, canonical_entities, canonical_keys, threshold=90):
    """Normalize entities using NER and fuzzy matching."""

    if not text:
        return ""
    
    """ Step 1: NER-based normalization - Replace recognized person or organization entities in the text 
     with their canonical forms defined in entity_dict"""
    doc = nlp_ner(text)
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG"] and ent.text in canonical_entities:
            text = re.sub(r'\b{}\b'.format(re.escape(ent.text)), canonical_entities[ent.text], text)
    
    """Step 2: Fuzzy matching for remaining words, replace those that closely match 
    canonical entity names based on a similarity threshold, correcting typos or variations"""
    normalized_words = []
    for word in text.split():
        match, score, _ = process.extractOne(word, canonical_keys, scorer=fuzz.ratio)
        normalized_words.append(canonical_entities[match] if score >= threshold else word)
    
    return " ".join(normalized_words)

# ---------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------Tokenization-----------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

def spacy_tokenize(text):
    doc = nlp(text)
    tokens = []
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG"]:
            tokens.append(ent.text.replace(" ", "_"))
    tokens += [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return tokens

# ---------------------------------------------------------------------------------------------------------------------------------
# ------------------------------Topic Modeling Utilities---------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

def topic_diversity(topics):
    """Compute diversity of top words across topics"""
    top_words = [w.split('*"')[1].replace('"', '') for t in topics for w in t[1].split(' + ')]
    unique_words = len(set(top_words))
    return unique_words / len(top_words) if top_words else np.nan

def topic_vector(topics, dictionary):
    """Convert topic to vector (bag-of-words) for stability"""
    vec = np.zeros(len(dictionary))
    for t in topics:
        for term in t[1].split(' + '):
            weight, word = term.split('*"')
            word = word.replace('"','')
            if word in dictionary.token2id:
                vec[dictionary.token2id[word]] = float(weight)
    return vec

# ---------------------------------------------------------------------------------------------------------------------------------
# ------------------------------Window-Based Processing----------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

def process_window(current_start, tweets, window_size, prev_topics_vecs=None):
    current_end = current_start + window_size
    mask = (tweets['created_at'] >= current_start) & (tweets['created_at'] < current_end)
    subset = tweets.loc[mask]

    # Dictionary & corpus
    global_dict = corpora.Dictionary(tweets['tokens'])
    global_dict.filter_extremes(no_below=10, no_above=0.4) # filtering words that appear too much and too little times
    dictionary = global_dict
    corpus = [dictionary.doc2bow(tokens) for tokens in subset['tokens']]

    # Train LDA
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=8,
        random_state=42,
        passes=10, 
        alpha='auto',
        chunksize=2000
    )

    # Coherence
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=subset['tokens'],
        dictionary=dictionary,
        coherence='c_v'
    )
    coherence = coherence_model.get_coherence()

    # Perplexity
    perplexity = lda_model.log_perplexity(corpus)

    # Topics
    topics = lda_model.print_topics(num_words=5)
    topics_str = [" | ".join([word for word in t[1].split(" + ")]) for t in topics]

    # Topic diversity
    diversity = topic_diversity(topics)

    # Topic stability
    stability = np.nan
    curr_topics_vecs = [topic_vector([t], dictionary) for t in topics]
    if prev_topics_vecs is not None:
        sims = []
        for vec1 in prev_topics_vecs:
            similarities = [cosine_similarity([vec1], [vec2])[0][0] for vec2 in curr_topics_vecs]
            sims.append(max(similarities))  # only keep best match
        stability = np.mean(sims) if sims else np.nan

    return {
        'start_date': current_start.date(),
        'end_date': current_end.date(),
        'num_docs': len(subset),
        'coherence': coherence,
        'perplexity': perplexity,
        'diversity': diversity,
        'stability': stability,
        'topics': topics_str,
        'topics_vecs': curr_topics_vecs  # keep for next window
    }
