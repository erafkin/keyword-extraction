from gensim.summarization import keywords

class GensimTextRankImpl:

    def __init__(self, text):
        self.text = text

    def getKeywords(self):
        return (keywords(self.text, pos_filter=('NN', 'NP'), lemmatize=True).split('\n'))

