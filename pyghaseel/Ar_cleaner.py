from tashaphyne.stemming import ArabicLightStemmer
import qalsadi.lemmatizer
import arabicstopwords.arabicstopwords as stp
import pyarabic.araby as araby
from functools import reduce

print("access ar cleaner")


class Ar_cleaner():
    def __init__(self):
        self.ar_stopwords = []
        self.my_stemmer = ArabicLightStemmer()
        self.my_lemmatizer = qalsadi.lemmatizer.Lemmatizer()
        self.stp = stp
    
    def setStopwords(self, ar_stopwords):
        self.ar_stopwords = ar_stopwords
    
    def clean_text(self, text, ar_lemm=False, ar_stem=False, fall_to_stem=False):
        text = self.removeMeaninglessWords(text)
        text = self.removeStopwords(text)
        
        for i in range(len(text)):
#             text[i] = text[i].strip("وال")
#             text[i] = text[i].strip("ال")
            original_word = text[i]
            text[i] = self.normalizeHamza(text[i])
            text[i] = self.stripTashkeel(text[i])
            
            text[i] = self.handleLastHaAndAlef(text[i], ar_lemm, ar_stem)
            text[i] = self.handleLastHa(text[i], ar_lemm, ar_stem)
            text[i] = self.cleanAlefLayenah(text[i], ar_lemm, ar_stem)
            
            text[i] = self.applyAllowedNormalizers(text[i], ar_lemm, ar_stem)
            text[i] = self.fallToStem(text[i], original_word, ar_lemm, ar_stem, fall_to_stem)
        
        text = self.removeMeaninglessWords(text)
        
        return text
    
    def applyAllowedNormalizers(self, word, ar_lemm, ar_stem):
        if ar_lemm:
            word = self.lemmatizer(word)
#             word = ISRIStemmer().suf32(word)
        if ar_stem:
            word = self.stemmer(word)
        return word

    def stemmer(self, word):
        return self.my_stemmer.light_stem(word) # stemming arabic text

    def lemmatizer(self, word):
        return self.my_lemmatizer.lemmatize(word) # lemmatizing arabic text

    def removeStopwords(self, text):
        return [word for word in text if not self.stp.is_stop(word)]

    def removeMeaninglessWords(self, text):
        return [word for word in text if word not in self.ar_stopwords]
    
    def normalizeHamza(self, word):
        return araby.normalize_hamza(word, method="tasheel")
    
    def stripTashkeel(self, text):
        return araby.strip_tashkeel(text)
        
    def cleanAlefLayenah(self, word, ar_lemm, ar_stem):
        return self.handleAlefLayenah(word, "ى", "ي", ar_lemm, ar_stem)
    
    def cleanHa(self, word, ar_lemm, ar_stem):
        return self.handleLastHa(word, ar_lemm, ar_stem)
    
    def handleSpecialCase(self, word, preferences):
        conditions = all(preferences["conditions"])
        if conditions:
            alt_word = word
            for func in preferences["alt_word"]:
                alt_word = func(alt_word)
            ref = self.checkEffect(word, alt_word, preferences["ar_lemm"], preferences["ar_stem"])
            return ref
        else:
            return word
    
    def handleAlefLayenah(self, word, char, alt_char, ar_lemm, ar_stem):
        preferences = {
            "conditions":["ى" in word],
            "alt_word":[lambda x: x.replace("ى", "ي")],
            "ar_lemm": ar_lemm,
            "ar_stem": ar_stem
        }
        return self.handleSpecialCase(word, preferences)
            
    def handleLastHa(self, word, ar_lemm, ar_stem):
        preferences = {
                "conditions":[word[-1] == "ه"],
                "alt_word":[lambda x: x[:-1]+"ة"],
                "ar_lemm": ar_lemm,
                "ar_stem": ar_stem
        }
        return self.handleSpecialCase(word, preferences)
        
    def handleLastHaAndAlef(self, word, ar_lemm, ar_stem):
        preferences = {
                "conditions":[word[-1] == "ه", "ى" in word],
                "alt_word":[lambda x: x[:-1]+"ة", lambda x: x.replace("ى", "ي")],
                "ar_lemm": ar_lemm,
                "ar_stem": ar_stem
        }
        return self.handleSpecialCase(word, preferences)
    
    def checkEffect(self, word, alt_word, ar_lemm, ar_stem):
        NotNormalizable = self.doesAllowedCleanersEffect(word, ar_lemm, ar_stem)
        Normalizable = self.doesAllowedCleanersEffect(alt_word, ar_lemm, ar_stem)
        
        if not NotNormalizable and Normalizable:
            return alt_word
        else:
            return word
    
    def doesAllowedCleanersEffect(self, word, ar_lemm, ar_stem):
#         if word[:3] == "وال":
#             word = word[3:]
#         elif word[:2] == "ال":
#             word = word[2:]
        
        return False if self.applyAllowedNormalizers(word, ar_lemm, ar_stem) == word else True
    
    def fallToStem(self, word, alt_word, ar_lemm, ar_stem, fall_to_stem):
        if word == alt_word and fall_to_stem:
            return self.stemmer(word)
        else:
            return word
