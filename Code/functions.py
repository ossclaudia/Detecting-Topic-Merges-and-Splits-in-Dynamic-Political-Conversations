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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

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
    urls = re.sub(r"/", " ", urls)             # substitute / for spaces
    urls = re.sub(r"\s+", " ", urls).strip()   # substitute numerous consecutive spaces for 1 and strips the beggining and end   
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
    combined = emoji.replace_emoji(combined, replace='') # remove all emojis
    combined = re.sub(r'http\S+|www\S+', '', combined) # remove urls   
    combined = re.sub(r'#', '', combined) # remove #
    combined = re.sub(r'@', '', combined) # Maintain the person's account name, just taking the @ out
    combined = re.sub(r'[^a-zA-Z\s]', ' ', combined) # Removes all non-letter characters (numbers, punctuation, special symbols)
    combined = combined.lower() # Converts all text to lowercase
    combined = re.sub(r'\s+', ' ', combined).strip() # Replaces multiple spaces, tabs, or newlines with a single space

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

    tokens += [token.text for token in doc if token.is_alpha]

    return tokens

# ---------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------LDA------------------------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------NMF------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

def nmf_topic_diversity(topics_words):
    """Compute diversity of top words across NMF topics"""
    all_words = [w for topic in topics_words for w in topic]
    unique_words = len(set(all_words))
    return unique_words / len(all_words) if all_words else np.nan

def nmf_topic_vectors(H):
    """Each row of H is a topic vector"""
    return [h for h in H]

def process_window_nmf(current_start, tweets, window_size, global_vocab, prev_topics_vecs=None, n_topics=8):
    current_end = current_start + window_size
    mask = (tweets['created_at'] >= current_start) & (tweets['created_at'] < current_end)
    subset = tweets.loc[mask]

    if len(subset) < 5:
        return None

    docs = [" ".join(tokens) for tokens in subset['tokens']]

    # Use the fixed global vocabulary
    vectorizer = TfidfVectorizer(vocabulary=global_vocab)
    tfidf = vectorizer.fit_transform(docs)
    vocab = vectorizer.get_feature_names_out()

    nmf_model = NMF(n_components=n_topics, random_state=42, max_iter=1000)
    W = nmf_model.fit_transform(tfidf)
    H = nmf_model.components_

    # Top words
    topics_words = []
    for topic_vec in H:
        top_indices = topic_vec.argsort()[-10:][::-1]
        top_words = [vocab[i] for i in top_indices]
        topics_words.append(top_words)
    topics_str = [" | ".join(words) for words in topics_words]

    # Coherence
    dictionary = corpora.Dictionary(subset['tokens'])
    coherence_model = CoherenceModel(
        topics=topics_words,
        texts=subset['tokens'],
        dictionary=dictionary,
        coherence='c_v'
    )
    coherence = coherence_model.get_coherence()

    # Diversity
    diversity = nmf_topic_diversity(topics_words)

    # Stability
    stability = np.nan
    curr_topics_vecs = [h for h in H]
    if prev_topics_vecs is not None:
        sims = []
        for vec1 in prev_topics_vecs:
            similarities = [cosine_similarity([vec1], [vec2])[0][0] for vec2 in curr_topics_vecs]
            sims.append(max(similarities))
        stability = np.mean(sims) if sims else np.nan

    reconstruction_err = nmf_model.reconstruction_err_

    return {
        'start_date': current_start.date(),
        'end_date': current_end.date(),
        'num_docs': len(subset),
        'coherence': coherence,
        'reconstruction_error': reconstruction_err,
        'diversity': diversity,
        'stability': stability,
        'topics': topics_str,
        'topics_vecs': curr_topics_vecs
    }

# ---------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------Ploting model's metrics----------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

def plot_topic_metrics(df_results, model_name="Model"):
    """
    Plots coherence, stability, diversity, and optionally reconstruction_error or perplexity 
    for a given model results DataFrame.

    Displays up to 4 metrics in a 2x2 grid for better readability.
    """
    df = df_results.copy()
    df['start_date'] = pd.to_datetime(df['start_date'])
    
    # Detect optional metric
    extra_metric = None
    if 'reconstruction_error' in df.columns:
        extra_metric = 'reconstruction_error'
        extra_label = 'Reconstruction Error'
    elif 'perplexity' in df.columns:
        extra_metric = 'perplexity'
        extra_label = 'Perplexity'
    
    base_metrics = ['coherence', 'stability', 'diversity']
    metrics = base_metrics + ([extra_metric] if extra_metric else [])
    titles = ['Topic Coherence', 'Topic Stability', 'Topic Diversity'] + ([extra_label] if extra_metric else [])
    
    # Determine grid shape
    n_plots = len(metrics)
    n_cols = 2
    n_rows = (n_plots + 1) // 2  # round up
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2 * n_rows), sharex=True)
    axes = axes.flatten()  # make iterable

    for ax, metric, title in zip(axes, metrics, titles):
        if metric not in df.columns:
            ax.axis('off')
            continue
        ax.plot(df['start_date'], df[metric], marker='o', markersize=2, linewidth=1.2)
        ax.set_title(f"{model_name}: {title}", fontsize=13, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel(metric.replace('_', ' ').capitalize())
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    
    # Hide unused subplots (if odd number of metrics)
    for i in range(len(metrics), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()