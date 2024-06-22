import streamlit as st
import pickle
import stanza
from gensim.models import Word2Vec
from numpy import dot
from numpy.linalg import norm

# Define the cosine similarity function
def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# Load the Telugu language model
stanza.download('te')
nlp = stanza.Pipeline('te', processors='tokenize,pos,lemma,depparse')

# Load the Word2Vec model and triplets from the pickle files
word2vec_model = Word2Vec.load('word2vec_model (3).pkl')

with open('triplets (3).pkl', 'rb') as f:
    all_triplets = pickle.load(f)

# Define the tokenize_telugu function
def tokenize_telugu(text, nlp):
    # Assuming Stanza will handle tokenization
    doc = nlp(text)
    tokens = [word.text for sent in doc.sentences for word in sent.words]
    return tokens

# Define the answer_question_with_embeddings function
def answer_question_with_embeddings(question, triplets, word2vec_model, nlp):
    # Preprocess the question and get its word embeddings
    question_tokens = tokenize_telugu(question, nlp)
    question_embeddings = [word2vec_model.wv[token] for token in question_tokens if token in word2vec_model.wv]

    if not question_embeddings:
        return "Apologies, I couldn't understand the question."

    # Calculate cosine similarity between question embeddings and triplet embeddings
    max_similarity = -1
    best_triplet = None

    for triplet in triplets:
        triplet_embeddings = [word2vec_model.wv[token] for token in triplet if token in word2vec_model.wv]
        if not triplet_embeddings:
            continue

        # Calculate the similarity score between the question and the triplet
        total_similarity = sum(max(cosine_similarity(q_emb, t_emb) for t_emb in triplet_embeddings) for q_emb in question_embeddings)

        if total_similarity > max_similarity:
            max_similarity = total_similarity
            best_triplet = triplet

    # Formulate answer based on the best triplet
    if best_triplet:
        subject, relation, obj = best_triplet
        return f"{subject} {relation} {obj}"
    else:
        return "Apologies, I couldn't find an answer."

# Create the Streamlit app
st.title("Telugu Question Answering App")
st.write("Enter your Telugu question:")

question = st.text_input("")

if st.button("Get Answer"):
    answer = answer_question_with_embeddings(question, all_triplets, word2vec_model, nlp)
    st.write("Answer:", answer)
