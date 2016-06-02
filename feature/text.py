import re

from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer(language='russian')

class SentenceTokenzier:
    delimeter = re.compile('((?:(?<![0-9])\.|[!?\n])+)\W*')
    def tokenize(self, txt):
        lines = self.delimeter.split(txt)
        # assert len(lines) % 2 == 0
        sents = []
        for a, b in zip(lines[::2], lines[1::2]):
            sents.append(a+b)
        if len(lines) % 2 == 1 and lines[-1] != '':
            sents.append(lines[-1])
        return sents

sentence_tokenzier = SentenceTokenzier()

class WordTokenizer:
    pattern = re.compile('\s*(\w+)\s*')
    def tokenize(self, txt):
        txt = txt.lstrip()
        return list(filter(None, self.pattern.split(txt)))

word_tokenizer = WordTokenizer()

def preprocess_sentence(sent):
    if not isinstance(sent, str):
        return [], []
    tokens = word_tokenizer.tokenize(sent)
    tokens_stemmed = [stemmer.stem(x) for x in tokens]
    return tokens, tokens_stemmed


def preprocess(txt):
    if not isinstance(txt, str):
        return [], []
    sents = []
    sents_stemmed = []
    for sent in sentence_tokenzier.tokenize(txt):
        tokens, tokens_stemmed = preprocess_sentence(sent)
        sents.append(tokens)
        sents_stemmed.append(tokens_stemmed)
    return sents, sents_stemmed