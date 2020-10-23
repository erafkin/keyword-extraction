from rake_nltk import Rake

class RakeImpl:

    def __init__(self, text):
        self.text = text
        self.rake = Rake()

    def getKeywords(self):
        self.rake.extract_keywords_from_text(self.text)
        return self.rake.get_ranked_phrases()