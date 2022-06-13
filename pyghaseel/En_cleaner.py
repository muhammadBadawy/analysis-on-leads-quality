import nltk
from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')

print("access en cleaner")

snowball_stemmer = SnowballStemmer('english')

STOPWORDS = set(stopwords.words('english'))

class En_cleaner():
    def __init__(self):
        self.en_stopwords = []
    
    def setStopwords(self, en_stopwords):
        self.en_stopwords = en_stopwords
        
    def clean_text(self, text, en_lemm=False, en_stem=False):
        text =  self.remove_stopwords(text)
        
        if en_lemm:
            text = self.lemmatizer(text)
        if en_stem:
            text = self.stemmer(text)
        
        text = self.removeMeaninglessWords(text)
        return text

    def stemmer(self, text):
        return [snowball_stemmer.stem(word) for word in text]

    def lemmatizer(self, text):
        return [WordNetLemmatizer().lemmatize(word) for word in text]

    def remove_stopwords(self, text):
        return [word for word in text if word not in STOPWORDS]
    
    def removeMeaninglessWords(self, text):
        return [word for word in text if word not in self.en_stopwords]
