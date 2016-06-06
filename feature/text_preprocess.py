#!/usr/bin/env python3
import re

from nltk.stem import SnowballStemmer
from nltk.tokenize.treebank import TreebankWordTokenizer

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

class WordTokenizer(TreebankWordTokenizer):
    PUNCTUATION = [
        (re.compile(r'([:,])([^\d])'), r' \1 \2'),
        (re.compile(r'([:,])$'), r' \1 '),
        (re.compile(r'\.\.\.'), r' ... '),
        (re.compile(r'[;@#$%&]'), r' \g<0> '),
        (re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'), r'\1 \2\3 '),
        (re.compile(r'[?!]+'), r' \g<0> '),
        (re.compile(r"([^'])' "), r"\1 ' "),
    ]

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

if __name__ == '__main__':
    import argparse
    import sys
    import json
    from collections import OrderedDict
    jsonify = lambda x: json.dumps(x, ensure_ascii=False)

    arg_parser = argparse.ArgumentParser(description='preprocess title and description')
    arg_parser.add_argument('--keep-raw', help='keep raw text', action='store_true')
    args = arg_parser.parse_args()

    for line in sys.stdin:
        line = json.loads(line.rstrip())
        title, title_stemmed = preprocess(line['title'])
        description, description_stemmed = preprocess(line['description'])
        res = OrderedDict([
            ('itemID', line['itemID']),
            ('title', title),
            ('title_stemmed', title_stemmed),
            ('description', description),
            ('description_stemmed', description_stemmed)
        ])
        if args.keep_raw:
            res.update([('title_raw', line['title']),
                        ('description_raw', line['description'])])
        print(jsonify(res))
