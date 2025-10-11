# ---------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------Importing Packages----------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

import re
import string
import html
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
from datetime import timedelta
from itertools import product

# ---------------------------------------------------------------------------------------------------------------------------------
# --------------------------------Text Cleaning Utilities--------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

def replace_html(text):
    return html.unescape(text) if isinstance(text, str) else ""

def clean_hashtags(hashtags):
    if pd.isna(hashtags):
        return ""
    hashtags = re.sub(r"#", "", hashtags)
    hashtags = re.sub(r",", " ", hashtags)
    return hashtags.strip()

def clean_urls(urls):
    if pd.isna(urls):
        return ""
    urls = re.sub(r"https?://", "", urls)
    urls = re.sub(r"www\.", "", urls)
    urls = re.sub(r"[^a-zA-Z/]", " ", urls)
    urls = re.sub(r"/", " ", urls)
    urls = re.sub(r"\s+", " ", urls).strip()
    irrelevant = {"com", "org", "gov", "net", "html"}
    words = [w for w in urls.split() if w not in irrelevant]
    return " ".join(words)

def clean_tweet_row(row):
    text, hashtags, urls = row['text'], row['hashtags'], row['urls']
    text = replace_html(text)
    hashtags_clean = clean_hashtags(hashtags)
    url_words = clean_urls(urls)
    combined = f"{text} {hashtags_clean} {url_words}"
    combined = re.sub(r'\bRT\b', '', combined)
    combined = emoji.replace_emoji(combined, replace='')
    combined = re.sub(r'http\S+|www\S+', '', combined)
    combined = re.sub(r'#', '', combined)
    combined = re.sub(r'@', '', combined)
    combined = re.sub(r'[^a-zA-Z\s]', ' ', combined)
    combined = combined.lower()
    combined = re.sub(r'\s+', ' ', combined).strip()
    tokens = [word for word in combined.split() if word not in stop_words]
    doc = nlp(" ".join(tokens))
    lemmas = [token.lemma_ for token in doc]
    return " ".join(lemmas)

# ---------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------Parallel Processing----------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

def parallel_apply(df, func):
    with Pool(cpu_count()) as p:
        result = p.map(func, [row for _, row in df.iterrows()])
    return result

# ---------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------Entity Normalization---------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

def normalize_entities(text, canonical_entities, canonical_keys, threshold=90):
    if not text:
        return ""
    doc = nlp_ner(text)
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG"] and ent.text in canonical_entities:
            text = re.sub(r'\b{}\b'.format(re.escape(ent.text)), canonical_entities[ent.text], text)
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
    tokens = [ent.text.replace(" ", "_") for ent in doc.ents if ent.label_ in ["PERSON", "ORG"]]
    tokens += [token.text for token in doc if token.is_alpha]
    return tokens

# ---------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------Basic Metrics--------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

def topic_diversity(topics_words):
    all_words = [w for topic in topics_words for w in topic]
    unique_words = len(set(all_words))
    return unique_words / len(all_words) if all_words else np.nan

def topic_coherence(topics_words, texts):
    dictionary = corpora.Dictionary(texts)
    coherence_model = CoherenceModel(
        topics=topics_words,
        texts=texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    return coherence_model.get_coherence()

# ---------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------Temporal Metrics-----------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

def temporal_topic_smoothness(prev_vecs, curr_vecs):
    if prev_vecs is None or curr_vecs is None:
        return np.nan
    sims = []
    for vec1 in prev_vecs:
        similarities = [cosine_similarity([vec1], [vec2])[0][0] for vec2 in curr_vecs]
        sims.append(max(similarities))
    return np.mean(sims) if sims else np.nan

def temporal_topic_coherence(prev_topics_words, curr_topics_words):
    if prev_topics_words is None or curr_topics_words is None:
        return np.nan
    overlaps = []
    for t1 in prev_topics_words:
        for t2 in curr_topics_words:
            inter = len(set(t1) & set(t2))
            union = len(set(t1) | set(t2))
            if union > 0:
                overlaps.append(inter / union)
    return np.mean(overlaps) if overlaps else np.nan

# ---------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------LDA------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

def train_lda(subset, n_topics=8, dictionary=None):
    if dictionary is None:
        raise ValueError("A global dictionary must be provided for LDA")
    corpus = [dictionary.doc2bow(tokens) for tokens in subset['tokens']]
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        random_state=42,
        passes=10,
        alpha='auto',
        chunksize=2000
    )
    topics_words = []
    topic_vecs = []
    for topic in lda_model.show_topics(num_topics=n_topics, num_words=len(dictionary), formatted=False):
        words = [w for w, _ in topic[1]]
        topics_words.append(words)
        vec = np.zeros(len(dictionary))
        for word, weight in topic[1]:
            if word in dictionary.token2id:
                vec[dictionary.token2id[word]] = weight
        topic_vecs.append(vec)
    coherence = topic_coherence(topics_words, subset['tokens'])
    diversity = topic_diversity(topics_words)
    return topics_words, topic_vecs, coherence, diversity

# ---------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------NMF------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

def train_nmf(subset, global_vocab, n_topics=8):
    docs = [" ".join(tokens) for tokens in subset['tokens']]
    if len(docs) < 5:
        return None
    vectorizer = TfidfVectorizer(vocabulary=global_vocab)
    tfidf = vectorizer.fit_transform(docs)
    vocab = vectorizer.get_feature_names_out()
    nmf_model = NMF(n_components=n_topics, random_state=42, max_iter=1000)
    W = nmf_model.fit_transform(tfidf)
    H = nmf_model.components_
    topics_words = []
    topic_vecs = []
    for topic_vec in H:
        top_indices = topic_vec.argsort()[-10:][::-1]
        top_words = [vocab[i] for i in top_indices]
        topics_words.append(top_words)
        topic_vecs.append(topic_vec)
    coherence = topic_coherence(topics_words, subset['tokens'])
    diversity = topic_diversity(topics_words)
    return topics_words, topic_vecs, coherence, diversity

# ---------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------Sliding Window----------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

def process_window(model_type, start_date, tweets, window_size, global_vocab=None,
                   prev_topics=None, prev_vecs=None, n_topics=8, dictionary=None):
    current_end = start_date + window_size
    mask = (tweets['created_at'] >= start_date) & (tweets['created_at'] < current_end)
    subset = tweets.loc[mask]
    if len(subset) < 10:
        return None
    if model_type == 'lda':
        topics, vecs, coherence, diversity = train_lda(subset, n_topics, dictionary=dictionary)
    elif model_type == 'nmf':
        topics, vecs, coherence, diversity = train_nmf(subset, global_vocab, n_topics)
        if topics is None:
            return None
    else:
        raise ValueError("model_type must be 'lda' or 'nmf'")
    tts = temporal_topic_smoothness(prev_vecs, vecs)
    ttc = temporal_topic_coherence(prev_topics, topics)
    ttq = np.nan if np.isnan(tts) or np.isnan(ttc) else tts * ttc
    tq = coherence * diversity
    dtq = np.nan if np.isnan(ttq) else 0.5 * (tq + ttq)
    return {
        'start_date': start_date.date(),
        'end_date': current_end.date(),
        'num_docs': len(subset),
        'coherence': coherence,
        'diversity': diversity,
        'tq': tq,
        'tts': tts,
        'ttc': ttc,
        'ttq': ttq,
        'dtq': dtq,
        'topics': [" | ".join(t) for t in topics],
        'topic_vecs': vecs,
        'topic_words': topics
    }


# ---------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------Execution------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

def run_dynamic_pipeline(tweets, model_type='lda', n_topics=8,
                         window_days=7, step_days=3):
    start_date = tweets['created_at'].min()
    end_date = tweets['created_at'].max()
    window_size = timedelta(days=window_days)
    step_size = timedelta(days=step_days)

    global_vocab = None
    global_dictionary = None

    if model_type == 'nmf':
        vectorizer = TfidfVectorizer(max_df=0.4, min_df=10)
        _ = vectorizer.fit([" ".join(tokens) for tokens in tweets['tokens']])
        global_vocab = vectorizer.get_feature_names_out()
    elif model_type == 'lda':
        global_dictionary = corpora.Dictionary(tweets['tokens'])
        global_dictionary.filter_extremes(no_below=10, no_above=0.4)

    current_start = start_date
    results, prev_topics, prev_vecs = [], None, None

    while current_start + window_size <= end_date:
        res = process_window(
            model_type, current_start, tweets, window_size,
            global_vocab, prev_topics, prev_vecs, n_topics, dictionary=global_dictionary
        )
        if res:
            prev_topics = res['topic_words']
            prev_vecs = res['topic_vecs']
            results.append(res)
        current_start += step_size

    df = pd.DataFrame(results)
    df.drop(columns=['topic_vecs', 'topic_words'], inplace=True, errors='ignore')
    return df
