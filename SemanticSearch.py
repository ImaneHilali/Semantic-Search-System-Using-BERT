import os
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import sqlite3
from tika import parser
import re
from concurrent.futures import ProcessPoolExecutor

STOPWORDS_LANGUAGE = 'english'
MAX_TOKEN_LENGTH = 512
BATCH_SIZE = 10

nltk.download('punkt')
nltk.download('stopwords')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

db_file = "files.db"
with sqlite3.connect(db_file) as conn:
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS files (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        path TEXT NOT NULL,
                        content TEXT NOT NULL
                    )''')
    conn.commit()

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation]
    stop_words = set(stopwords.words(STOPWORDS_LANGUAGE))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(tokens)

def remove_space(text):
    return re.sub(r'\s+', ' ', text).strip()

def convert_to_txt(input_file_path):
    try:
        parsed_content = parser.from_file(input_file_path)
        text_content = parsed_content.get('content', '')
        return remove_space(text_content)
    except Exception as e:
        print(f"Error converting file '{input_file_path}' to TXT: {e}")
        return None

def insert_file_to_db(name, path, content):
    cursor.execute("INSERT INTO files (name, path, content) VALUES (?, ?, ?)", (name, path, content))
    conn.commit()

def retrieve_files_from_db():
    cursor.execute("SELECT id, name, path, content FROM files")
    return cursor.fetchall()

def process_files_in_batches(data_folder, start_index, end_index):
    batch_files = os.listdir(data_folder)[start_index:end_index]
    for filename in batch_files:
        filepath = os.path.join(data_folder, filename)
        if os.path.isfile(filepath) and not any(file_info[1] == filename for file_info in retrieve_files_from_db()):
            text_content = convert_to_txt(filepath)
            if text_content:
                insert_file_to_db(filename, filepath, text_content)

def semantic_search(query, min_similarity_score=0):
    preprocessed_query = preprocess_text(query)
    query_tokens = tokenizer.encode(preprocessed_query, add_special_tokens=True, max_length=MAX_TOKEN_LENGTH, truncation=True, padding='max_length', return_tensors='pt')

    with torch.no_grad():
        query_outputs = model(query_tokens)
        query_embedding = query_outputs.last_hidden_state[:, 0, :].numpy()

    query_word_freq = {word: preprocessed_query.split().count(word) for word in preprocessed_query.split()}

    similarity_scores = []
    records = retrieve_files_from_db()

    for file_info in records:
        preprocessed_text = preprocess_text(file_info[1] + ' ' + file_info[3])
        relevance_score = sum(query_word_freq.get(word, 0) for word in preprocessed_text.split()) / len(preprocessed_query.split())

        document_tokens = tokenizer.encode(preprocessed_text, add_special_tokens=True, max_length=MAX_TOKEN_LENGTH, truncation=True, padding='max_length', return_tensors='pt')
        with torch.no_grad():
            embeddings = model(document_tokens).last_hidden_state[:, 0, :].numpy()
        cosine_sim = cosine_similarity(query_embedding, embeddings.reshape(1, -1)).flatten()[0]
        similarity_score = cosine_sim * relevance_score

        similarity_scores.append(similarity_score)

    top_indices = np.argsort(similarity_scores)[::-1][:10]
    return [(records[i][2], similarity_scores[i]) for i in top_indices]

def main():
    data_folder = "C:\\Support Document\\"
    total_files = len(os.listdir(data_folder))

    for batch_start in range(0, total_files, BATCH_SIZE):
        process_files_in_batches(data_folder, batch_start, min(batch_start + BATCH_SIZE, total_files))

    queries = [(f"Sample query {i}", 0) for i in range(total_files)]
    with ProcessPoolExecutor() as executor:
        search_results_batches = executor.map(semantic_search, queries)

    search_results = [result for batch in search_results_batches for result in batch]

    for i, (file_path, score) in enumerate(search_results, start=1):
        print(f"Result {i}:")
        print(f"Similarity Score: {score}")
        print(f"File Path: {file_path}\n")

    conn.close()

if __name__ == "__main__":
    main()
