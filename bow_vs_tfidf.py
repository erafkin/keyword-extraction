# calculating tf-idf values 
from sklearn.feature_extraction.text import TfidfVectorizer 
import pandas as pd 
import nltk
import os

files = os.listdir('./dnc_speeches')
# iterate over the list getting each file 
speeches = []
for fle in files:
    # open the file and then call .read() to get the text 
    with open('./dnc_speeches/'+fle) as f:
        text = f.read()
        speeches.append(text)

# Creating the Bag of Words model  
word2count = {}  
for data in speeches:  
    words = nltk.word_tokenize(data)  
    for word in words:  
        if word not in word2count.keys():  
            word2count[word] = 1
        else:  
            word2count[word] += 1

tfidf = TfidfVectorizer(min_df = 2, max_df = 0.5, ngram_range = (1, 2)) 
features = tfidf.fit_transform(speeches) 
  
print(pd.DataFrame(features.todense(), columns = tfidf.get_feature_names()))
    