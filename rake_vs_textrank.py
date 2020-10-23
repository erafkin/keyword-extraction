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

# with open('./inslee_speech.txt') as f:
#     text = f.read()
#     for sentence in text.split("."):
#         sentences.append(sentence)
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

for keyword in keywords:
    print (keyword)
    search_terms = keyword
    doc_vectors = tfidf_vectorizer.fit_transform([search_terms] + newTexts)

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

