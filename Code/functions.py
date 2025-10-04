import pandas as pd
import re
import string
import emoji
import spacy
from nltk.corpus import stopwords
import nltk
#nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
stop_words = set(stopwords.words('english'))
import html
from multiprocessing import Pool, cpu_count
import spacy
from rapidfuzz import process, fuzz
import re
nlp_ner = spacy.load("en_core_web_sm")

def replace_html(text):
    """Expose HTML entities (e.g. &amp; -> &) so they can be cleaned later."""
    return html.unescape(text) if isinstance(text, str) else ""

def clean_hashtags(hashtags):
    """Remove # and commas from hashtags column."""
    if pd.isna(hashtags):
        return ""
    hashtags = re.sub(r"#", "", hashtags)
    hashtags = re.sub(r",", " ", hashtags)
    return hashtags.strip()

def clean_urls(urls):
    """Extract meaningful words from URLs and remove irrelevant tokens."""
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
    """Combine text, hashtags, and URL words, and clean the tweet."""
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

def parallel_apply(df, func):
    with Pool(cpu_count()) as p:
        result = p.map(func, [row for _, row in df.iterrows()])
    return result

def normalize_entities(text, canonical_entities, canonical_keys, threshold=90):
    """
    Normalize entities using NER and fuzzy matching.
    
    threshold (int): min similarity for fuzzy match (0-100)
    
    """

    if not text:
        return ""
    
    # Step 1: NER-based normalization - Replace recognized person or organization entities in the text 
    # with their canonical forms defined in entity_dict
    doc = nlp_ner(text)
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG"]:
            if ent.text in canonical_entities:
                # replace whole word occurrences
                text = re.sub(r'\b{}\b'.format(re.escape(ent.text)), canonical_entities[ent.text], text)
    
    # Step 2: Fuzzy matching for remaining words For remaining words, replace those that closely match 
    # canonical entity names based on a similarity threshold, correcting typos or variations
    words = text.split()
    normalized_words = []
    for word in words:
        match, score, _ = process.extractOne(word, canonical_keys, scorer=fuzz.ratio)
        if score >= threshold:
            normalized_words.append(canonical_entities[match])
        else:
            normalized_words.append(word)
    
    return " ".join(normalized_words)
