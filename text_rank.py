# heavily influenced by https://towardsdatascience.com/textrank-for-keyword-extraction-by-python-c0bae21bcec0
from collections import OrderedDict
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
parser = spacy.load('en_core_web_sm')

class TextRankImpl():
    def __init__(self):
        ## can play around with these values
        self.d = 0.85 # damping coefficient, usually is .85
        self.min_diff = 1e-5 # convergence threshold --> bigger is faster, smaller is better.
        self.epochs = 10 # iteration steps
        self.node_weight = None # save keywords and its weight

    def my_stopwords(self, stopwords):
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = parser.vocab[word]
            lexeme.is_stop = True
    
    def sentence_segment(self, doc, pos, lower):
        # only store words that are are of the right POS
        sentences = []
        for sent in doc.sents:
            words = []
            for token in sent:
                if token.pos_ in pos and token.is_stop == False:
                    if lower is True:
                        words.append(token.text.lower())
                    else:
                        words.append(token.text)
            sentences.append(words)
        return sentences
    
    ## indexes words
    def get_vocab(self, sentences):
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab
    
    ## we look at all words that are close to each other in a sentence based on a certain "chunk/window size"
    ## any two words within a certain chunk size have an undirected edge in the graph. 
    ## these are pairs with connections so that we can put it in the graph.
    def get_token_pairs(self, chunk_size, sentences):
        pairs = []
        for sentence in sentences:
            for i in range(len(sentence)):
                for j in range(i+1, i+chunk_size):
                    if j>= len(sentence):
                        break
                    pair = (sentence[i], sentence[j])
                    if pair not in pairs:
                        pairs.append(pair)
        return pairs
    
    # make the matrix symmetrical
    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())
    
    def get_matrix(self, vocab, pairs):
        vocab_size = len(vocab)
        matrix = np.zeros((vocab_size, vocab_size),dtype = 'float')
        for word1, word2 in pairs:
            i, j = vocab[word1], vocab[word2]
            matrix[i,j] = 1
        
        # make her symmetric
        matrix = self.symmetrize(matrix)

        # Normalize the matrix
        norm = np.sum(matrix, axis = 0)
        matrix_norm = np.divide(matrix, norm, where=norm!=0) # make sure to ignore 0's in norm

        return matrix_norm

    def analyze(self, text, pos=['NOUN', 'PROPN', 'VERB'], chunk_size=4, lower=False, stopwords=list()):
        #set the stop words. maybe add to this "election" etc.
        self.my_stopwords(stopwords)

        # parse the text using spaCy
        document = parser(text)

        # get sentences
        sentences = self.sentence_segment(document, pos, lower)

        # build vocab dict
        vocab = self.get_vocab(sentences)

        # get token pairs
        token_pairs = self.get_token_pairs(chunk_size, sentences)

        # get normalized matrix
        matrix = self.get_matrix(vocab, token_pairs)

        # initialize for textrank value
        tr = np.array([1] * len(vocab))

        # iterate
        previous_tr = 0
        for epoch in range(self.epochs):
            tr = (1-self.d) + self.d * np.dot(matrix, tr)
            if abs(previous_tr - sum(tr))  < self.min_diff:
                break
            else:
                previous_tr = sum(tr)

        # get the weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = tr[index]
        
        self.node_weight = node_weight



    def get_keywords(self, number=10):
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1]))
        for i, (key, value) in enumerate(node_weight.items()):
            print(key + ' - ' + str(value))
            if i > number:
                break
    
