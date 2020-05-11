import itertools
import re
import string

import fasttext
import spacy
from spacy.tokens.doc import Doc

from pycountry import languages

class Preprocessing():
    def __init__(self, documents):
        
        url_pattern = "^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9_]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$"
        domain_pattern = "(?:[a-zA-Z0-9](?:[a-z0-9-_]{0,61}[a-z0-9])?\.)+[a-z0-9][a-z0-9-]{0,61}[a-z0-9]"
        
        spacy.tokens.Token.set_extension("is_email", getter=lambda token: len(re.findall("\S+@\S+", token.text)) > 0 , force=True)
        spacy.tokens.Token.set_extension("is_url", getter=lambda token: len(re.findall(url_pattern, token.text)) > 0, force=True)
        spacy.tokens.Token.set_extension("is_domain", getter=lambda token: len(re.findall(domain_pattern, token.text)) > 0, force=True)

        spacy.tokens.Doc.set_extension("has_email", getter=lambda doc: len(re.findall("\S+@\S+", doc.text)) > 0, force=True)
        spacy.tokens.Doc.set_extension("has_url", getter=lambda doc: len(re.findall(url_pattern, doc.text)) > 0, force=True)
        spacy.tokens.Doc.set_extension("has_domain", getter=lambda doc: len(re.findall(domain_pattern, doc.text)) > 0, force=True)
        spacy.tokens.Doc.set_extension("language", getter=lambda doc: self.detect_language(doc), force=True)

        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe(self.custom_cleaner, name="custom cleaner")

        PRETRAINED_MODEL_PATH = '/tmp/lid.176.bin'
        self.lang_model = fasttext.load_model(PRETRAINED_MODEL_PATH)

        self.documents = documents

    def adjust_characters(self, doc):
        words = [token.text for token in doc]
        words = [re.sub("[.]+", ". ", word).split() for word in words]
        words = itertools.chain(words)
        words = [re.sub("[-]+", "- ", word).split() for word in words]
        words = [re.sub("[_]+", " ", word).split() for word in words]
        return Doc(vocab=doc.vocab, words=words)

    def remove_special_chars(self, doc):
        words = [token.text for token in doc]
        words = [re.sub(f"[^A-Za-z0-9{string.punctuation}]+", "", word) for word in words]
        words = [word for word in words if word != ""]
        return Doc(vocab=doc.vocab, words=words)


    def remove_puctuation(self, doc):
        words = [re.sub(f"[{string.punctuation}]+", " \1 ", token.text) if not token._.is_email and not token._.is_url and not token._.is_domain else token.text for token in doc]
        words = ' '.join(words).split()
        words = [word for word in words if word != "\x01"]
        return Doc(vocab=doc.vocab, words=words)


    def remove_stopwords(self, doc):
        words = [token.text for token in doc if not token.is_stop]
        return Doc(vocab=doc.vocab, words=words)


    def apply_lemmatization(self, doc):
        words = [token.lemma_ for token in doc]
        return Doc(vocab=doc.vocab, words=words)


    def detect_language(self, doc):
        try:
            predictions = self.lang_model.predict(doc.text)
            lang = predictions[0][0].split("__")[-1]
            score = predictions[1][0]
            name = languages.get(alpha_2=lang).name
            return (name, lang, score)
        except Exception as e:
            return ("NA", "NA", 0)


    def custom_cleaner(self, doc):
        doc = self.remove_special_chars(doc)
        doc = self.remove_puctuation(doc)
        doc = self.remove_stopwords(doc)
        doc = self.apply_lemmatization(doc)
        return doc

    def run(self):
        return list(self.nlp.pipe(self.documents))
