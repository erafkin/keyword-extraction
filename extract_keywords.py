# from text_rank import TextRankImpl
from gensim_textrank import GensimTextRankImpl
import gensim.models
from rake import RakeImpl
import spacy
import os
from collections import OrderedDict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import contractions
from nltk.stem import SnowballStemmer, PorterStemmer
from nltk.corpus import stopwords
import string
import re
from sklearn.metrics.pairwise import linear_kernel
import gensim.downloader as api
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import SoftCosineSimilarity


def process_text(text):
    text = nltk.tokenize.casual.casual_tokenize(text)
    text = [each.lower() for each in text]
    text = [re.sub('[0-9]+', '', each) for each in text]
    text = [contractions.fix(each) for each in text]
    # text = [SnowballStemmer('english').stem(each) for each in text]
    text = [w for w in text if w not in string.punctuation]
    text = [w for w in text if w not in stop_words]
    text = [w for w in text if w not in ["america", "american", "nation", "elect", "look people", "know"]]
    text = [each for each in text if len(each) > 2]
    text = [each for each in text if ' ' not in each]
    return text

files = os.listdir('./dnc_speeches')
# iterate over the list getting each file 
speeches = []
speech_titles = []
for fle in files:
    # open the file and then call .read() to get the text 
    with open('./dnc_speeches/'+fle) as f:
        speech_titles.append(fle)
        text = f.read()
        speeches.append(text)
sentences = []

for speech in speeches:
    for sentence in speech.split("."):
        sentences.append(sentence)

stop_words = stopwords.words('english')

texts = [process_text(each) for each in speeches]
newTexts = []
for text in texts:
    newTexts.append(" ".join(text))

tfidf_vectorizer = TfidfVectorizer(max_df=0.5, min_df=0, max_features=2000, use_idf=True, tokenizer=process_text)
textRankImpl = GensimTextRankImpl(" ".join(newTexts))

keywords = textRankImpl.getKeywords()[:10]
print(keywords)


# Load the model: this is a big file, can take a while to download and open
glove = api.load("glove-wiki-gigaword-50")    
similarity_index = WordEmbeddingSimilarityIndex(glove)

search_terms = ["pollution"]
doc_vectors = tfidf_vectorizer.fit_transform(search_terms + newTexts)

# Build the term dictionary, TF-idf model
dictionary = Dictionary(texts+[search_terms])
tfidf = TfidfModel(dictionary=dictionary)
# Create the term similarity matrix.  
similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf)
# From: https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/soft_cosine_tutorial.ipynb
query_tf = tfidf[dictionary.doc2bow(search_terms)]

index = SoftCosineSimilarity(
            tfidf[[dictionary.doc2bow(document) for document in texts]],
            similarity_matrix)

doc_similarity_scores = index[query_tf]

# Output the sorted similarity scores and documents
sorted_indexes = np.argsort(doc_similarity_scores)[::-1]
for idx in sorted_indexes:
    print(f'{idx} \t {doc_similarity_scores[idx]:0.3f} \t {speech_titles[idx]}')

# Calculate cosine similarity
cosine_similarities = linear_kernel(doc_vectors[0:1], doc_vectors).flatten()
document_scores = [item.item() for item in cosine_similarities[1:]]
max_value = max(document_scores)
max_index = document_scores.index(max_value)
print(document_scores)
print(max_index)
print(speech_titles[max_index])

# idk how we want to use this but we should
rakeImpl = RakeImpl(" ".join(speeches))
print (rakeImpl.getKeywords()[:10])

