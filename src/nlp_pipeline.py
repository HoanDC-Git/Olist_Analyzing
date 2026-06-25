import os
import re
import time
import requests
import pandas as pd
import numpy as np
from config_loader import load_config

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def download_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

def clean_text(text, stopwords):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stopwords and len(w) > 2]
    return " ".join(words)

def translate_pt_to_en(text):
    """Native translation using Google Translate free endpoint with delays."""
    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {"client": "gtx", "sl": "pt", "tl": "en", "dt": "t", "q": text}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            result = response.json()
            return result[0][0][0].lower()
    except Exception as e:
        print(f"Translation failed for '{text}': {e}")
    return text

def extract_top_tfidf_terms(df, text_column, top_n=100):
    """Extract top N terms based on sum of TF-IDF scores."""
    # Use bigrams to capture 'não recomendo', 'não chegou', etc.
    tfidf = TfidfVectorizer(max_features=5000, min_df=10, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(df[text_column])
    
    # Sum TF-IDF scores for each word across all documents
    sum_tfidf = np.asarray(X_tfidf.sum(axis=0)).ravel()
    terms = tfidf.get_feature_names_out()
    
    # Get top N indices
    top_indices = sum_tfidf.argsort()[-top_n:][::-1]
    
    top_terms = []
    for idx in top_indices:
        top_terms.append({
            "term_pt": terms[idx],
            "score": sum_tfidf[idx]
        })
        
    return pd.DataFrame(top_terms)

def process_sentiment(df, sentiment_name, proc_dir):
    print(f"\nProcessing {sentiment_name} reviews ({len(df)} records)...")
    if len(df) == 0:
        return
        
    # Extract top TF-IDF terms
    top_terms_df = extract_top_tfidf_terms(df, "cleaned_text", top_n=100)
    
    print(f"Translating top 100 {sentiment_name} terms to English (with delay)...")
    translated_terms = []
    for idx, row in top_terms_df.iterrows():
        en_term = translate_pt_to_en(row["term_pt"])
        translated_terms.append(en_term)
        # Avoid rate limits
        time.sleep(0.3)
        
    top_terms_df["term_en"] = translated_terms
    
    output_path = os.path.join(proc_dir, f"nlp_tfidf_{sentiment_name}_top_words.csv")
    top_terms_df.to_csv(output_path, index=False)
    print(f"Saved top translated {sentiment_name} words to {output_path}")

def run_nlp_pipeline():
    print("--- Starting Semantic NLP Pipeline ---")
    config = load_config()
    proc_dir = config["paths"]["processed_data_dir"]
    reviews_path = os.path.join(proc_dir, "olist_order_reviews_dataset.csv")
    
    if not os.path.exists(reviews_path):
        print(f"Error: {reviews_path} not found.")
        return

    download_nltk_resources()
    from nltk.corpus import stopwords
    try:
        pt_stopwords = set(stopwords.words('portuguese'))
    except:
        pt_stopwords = set(["de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "é", "com", "não", "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como", "mas", "foi", "ao", "ele", "das", "tem", "à", "seu", "sua", "ou", "ser", "quando", "muito", "há", "nos", "já", "está", "eu", "também", "só", "pelo", "pela", "até", "isso", "ela", "entre", "era", "depois", "sem", "mesmo", "aos", "ter", "seus", "quem", "nas", "me", "esse", "eles", "estão", "você", "tinha", "foram", "essa", "num", "nem", "suas", "meu", "às", "minha", "têm", "numa", "pelos", "elas", "havia", "seja", "qual", "será", "nós", "tenho", "lhe", "deles", "essas", "esses", "pelas", "este", "fosse", "dele", "tu", "te", "vocês", "vos", "lhes", "meus", "minhas", "teu", "tua", "teus", "tuas", "nosso", "nossa", "nossos", "nossas", "dela", "delas", "esta", "estes", "estas", "aquele", "aquela", "aqueles", "aquelas", "isto", "aquilo", "estou", "estamos", "estão", "estive", "esteve", "estivemos", "estiveram", "estava", "estávamos", "estavam", "estivera", "estivéramos", "esteja", "estejamos", "estejam", "estivesse", "estivéssemos", "estivessem", "estiver", "estivermos", "estiverem", "hei", "há", "havemos", "hão", "houve", "houvemos", "houveram", "houvera", "houvéramos", "haja", "hajamos", "hajam", "houvesse", "houvéssemos", "houvessem", "houver", "houvermos", "houverem", "houverei", "houverá", "houveremos", "houverão", "houveria", "houveríamos", "houveriam", "sou", "somos", "são", "era", "éramos", "eram", "fui", "foi", "fomos", "foram", "fora", "fôramos", "seja", "sejamos", "sejam", "fosse", "fôssemos", "fossem", "for", "formos", "forem", "serei", "será", "seremos", "serão", "seria", "seríamos", "seriam", "tenho", "tem", "temos", "tém", "tinha", "tínhamos", "tinham", "tive", "teve", "tivemos", "tiveram", "tivera", "tivéramos", "tenha", "tenhamos", "tenham", "tivesse", "tivéssemos", "tivessem", "tiver", "tivermos", "tiverem", "terei", "terá", "teremos", "terão", "teria", "teríamos", "teriam"])

    # Remove negation words from stopwords so bigrams like 'nao recomendo' are preserved
    negation_words = {"não", "nao", "nem", "nunca", "jamais"}
    pt_stopwords = pt_stopwords - negation_words

    # General E-commerce noise to stopwords
    noise = ["produto", "comprei", "recebi", "loja", "prazo", "dia", "veio", "chegou", "entregue", "entrega", "pedido", "compra", "ainda", "agora", "aqui", "pois", "ter", "tudo", "nada", "porque", "bem", "muito", "antes", "super", "apenas", "veio", "chegou", "dar", "fazer", "ter", "ser", "ir", "poder", "estar"]
    pt_stopwords.update(noise)
    
    # Specific noise for Negative Reviews (Remove leaked positive words)
    pos_leakage_noise = ["bom", "gostei", "ótimo", "otimo", "excelente", "maravilhoso", "lindo", "perfeito", "recomendo", "adorei", "legal", "qualidade", "bom", "boa"]
    pt_stopwords_neg = pt_stopwords.copy()
    pt_stopwords_neg.update(pos_leakage_noise)

    df = pd.read_csv(reviews_path)
    
    # Positive Reviews (4-5 stars)
    pos_reviews = df[(df["review_score"] >= 4) & (df["review_comment_message"].notnull())].copy()
    print(f"Cleaning {len(pos_reviews)} positive review comments...")
    pos_reviews["cleaned_text"] = pos_reviews["review_comment_message"].apply(lambda x: clean_text(x, pt_stopwords))
    pos_reviews = pos_reviews[pos_reviews["cleaned_text"].str.strip() != ""]
    process_sentiment(pos_reviews, "positive", proc_dir)

    # Negative Reviews (1-2 stars)
    neg_reviews = df[(df["review_score"] <= 2) & (df["review_comment_message"].notnull())].copy()
    print(f"\nCleaning {len(neg_reviews)} negative review comments...")
    neg_reviews["cleaned_text"] = neg_reviews["review_comment_message"].apply(lambda x: clean_text(x, pt_stopwords_neg))
    neg_reviews = neg_reviews[neg_reviews["cleaned_text"].str.strip() != ""]
    process_sentiment(neg_reviews, "negative", proc_dir)

    print("\n--- Semantic NLP Pipeline Completed Successfully ---\n")

if __name__ == "__main__":
    run_nlp_pipeline()
