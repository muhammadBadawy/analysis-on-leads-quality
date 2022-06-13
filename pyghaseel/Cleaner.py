from .Ar_cleaner import Ar_cleaner
from .En_cleaner import En_cleaner
from .Singleton import Singleton
import re
import mysql.connector as sql
import pandas as pd

REPLACE_BY_SPACE_RE = re.compile('[/{}\|@;,]')
_S_ = re.compile("'s")
BAD_SYMBOLS_RE = re.compile('[^\[\]\',0-9a-zA-Z #+_\-,\'\u0621-\u064A\u0660-\u0669]')
SEMI_BAD_SYMBOLS_RE = re.compile(r' \(')
SINGLE_QUT_SYMBOLS_RE = re.compile(r"'\w")
NIGITAL_NUMBERS_RE = re.compile('[\d]')
SINGLE_CHAR_RE = re.compile(' . | .$')
ARABIC_CHARACTERS = re.compile(r"[\u0621-\u064A]+", re.S)
ENGLISH_SYMBOLS_RE = re.compile(r"[a-zA-Z]+", re.S)

class Cleaner(metaclass=Singleton):
    def __init__(self, db_configs=None):
        self.initiate()
        if db_configs:
            self.db_configs = db_configs
            self.updateStopWords()
    
    def clean(self, text, ar_lemm=False, en_lemm=False, ar_stem=False, en_stem=False, fall_to_stem=False, ar=True, en=False):
        text = str(text) # lowercase text
        text = text.lower() # lowercase text
        text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
        text = _S_.sub('', text) # delete 's from text
        text = SEMI_BAD_SYMBOLS_RE.sub(' ', text) # delete symbols which are in BAD_SYMBOLS_RE from text
        text = NIGITAL_NUMBERS_RE.sub('', text) # delete digital numbers which are in BAD_SYMBOLS_RE from text
        text = SINGLE_CHAR_RE.sub(' ', text) # delete single chars which are in BAD_SYMBOLS_RE from text
        
        result = []
        
        if ar:
            ar_text = re.findall(ARABIC_CHARACTERS, text)
            result += self.ar_cleaner.clean_text(ar_text, ar_lemm, ar_stem, fall_to_stem)
            del ar_text
        
        if en:
            en_text = re.findall(ENGLISH_SYMBOLS_RE, text)
            result += self.en_cleaner.clean_text(en_text, en_lemm, en_stem)
            del en_text
        
        return " ".join(result)
    
    def GeneralCleanerFunc(self, text):
        return self.clean(text, ar_lemm=True, en_lemm=True, fall_to_stem=True, ar=True, en=True) # 

    def lemm(self, text):
        return self.clean(text, ar_lemm=True, en_lemm=True, fall_to_stem=False, ar=True, en=True) # 

    def stem(self, text):
        return self.clean(text, ar_stem=True, en_stem=True, fall_to_stem=False, ar=True, en=True) # 

    def liteCleanerFunc(self, text):
        return self.clean(text, ar=False, en=False)
        
    def initiate(self):
        self.ar_cleaner = Ar_cleaner()
        self.en_cleaner = En_cleaner()
    
    def updateStopWords(self, new_configs=None):
        if new_configs:
            self.db_configs = new_configs
        
        host = self.db_configs["host"]
        user = self.db_configs["user"]
        password = self.db_configs["password"]
        db_name = self.db_configs["db_name"]
        table_name = self.db_configs["table_name"]
        word_column = self.db_configs["word_column"]
        language_column = self.db_configs["language_column"]
        ar_word_value = self.db_configs["ar_word_value"]
        en_word_value = self.db_configs["en_word_value"]
        
        conn = sql.connect(host=host, database=db_name, user=user, password=password)
        
        full_stopwords_df = pd.read_sql(f'SELECT {word_column}, {language_column} FROM {table_name};', conn)
        ar_stopwords = full_stopwords_df[full_stopwords_df[language_column] == ar_word_value][word_column].values
        en_stopwords = full_stopwords_df[full_stopwords_df[language_column] == en_word_value][word_column].values
        del full_stopwords_df
        
        conn.close()
        
        self.ar_cleaner.setStopwords(ar_stopwords)
        self.en_cleaner.setStopwords(en_stopwords)
