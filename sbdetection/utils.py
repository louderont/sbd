import re
import os
from typing import List, Dict
import nltk
import pathlib

class FileReader():

    def __init__(self, input: str, output: str) -> None:
        self.input = input
        self.output = output

    def generate_path(self, target, subfolder) -> str:
        return os.path.join(pathlib.Path(__file__).parent.resolve(), 'data', subfolder, target)

    def read(self) -> List[str]:
        path = self.generate_path(self.input, 'input')
        with open(path, 'r') as f:
            data = f.readlines()
        return data

    def write(self, data: List[str]) -> None:
        path = self.generate_path(self.output, 'output')

        with open(path, 'w') as f:
            for line in data:
                f.write(f'{line}\n')


class NltkCorpusReader():
    
    def __init__(self, corpus_name:str, from_sentence:int = 0, to_sentence:int = 300, extra_args:Dict = None) -> None:
        self.corpus_name = corpus_name
        self.from_sentence = from_sentence
        self.to_sentence = to_sentence
        self.sentence_sample = self.corpus_to_sentences(extra_args)

    @property
    def corpus_full_text(self):
        return ' '.join(self.sentence_sample)
    
    def corpus_stats(self, sentences):
        avg_sentence_length = round(sum([len(sent) for sent in sentences])/len(sentences))
        print(f'Corpus: {self.corpus_name}: len - {len(sentences)} / avg sent len - {avg_sentence_length}')
    
    def corpus_to_sentences(self, extra_args: Dict) -> List[str]:
        if not hasattr(nltk.corpus, self.corpus_name):
            nltk.download(self.corpus_name)
        corpus = getattr(nltk.corpus, self.corpus_name)
        if extra_args is None:
            sents = corpus.sents()
        else:
            sents = corpus.sents(**extra_args)
        raw_sentences = sents[self.from_sentence:self.to_sentence]
        sentences = [self._join_glue(sent) for sent in raw_sentences]
        
        self.corpus_stats(sentences)
        return sentences

    def _join_glue(self, words: List[str]) -> str:
        """Join a sentence, represented as a list of words into a single string
        
        As in nltk corpus, all tokens are separated, we must define a way to join 
        tokens into a sentence and respect the original grammar of the text. 
        
        So, we must:
            - remove space before dot, comma, questionmark.
            - remove spaces before and after dash, paranthesis and quotes.
              Removing leading and trailing space before quotes requires to know which quote are opening
              quotes and which one are closing quotes. It requires the text to use curly quotes annotation.  
        """
        
        PATTERN = """.,!:;"?'"""
        raw_sentence = sum([[word, " "] for word in words], [])
        sentence = []
        for i in range(0, len(raw_sentence)-1):
            if not(raw_sentence[i] == " " and raw_sentence[i+1] in PATTERN):
                sentence.append(raw_sentence[i])

        sentence = "".join(sentence)
        sentence = re.sub(r"( )(\.|\?|!|:|;|,|\)|''|-)", r'\2', sentence)
        sentence = re.sub(r"(\(|-|``)( )", r'\1', sentence)
        sentence = re.sub(r'``', r"''", sentence)
        
        # case of gutenberg
        sentence = re.sub(r"` ", "'", sentence)
        sentence = re.sub("\'", "'", sentence)
        return sentence
