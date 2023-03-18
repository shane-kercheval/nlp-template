from functools import cached_property, lru_cache, singledispatchmethod
import itertools
from typing import Callable, Collection, Iterable, Optional, Union, List

import pandas as pd
import numpy as np
import re
import regex
import textacy
import spacy
from spacy.language import Language
import spacy.lang.en.stop_words as sw
import spacy.tokenizer as stz
import spacy.tokens as st
import spacy.tokens.doc as sd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from helpsk.diff import diff_text

from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex

import source.library.regex_patterns as rp


STOP_WORDS_DEFAULT = sw.STOP_WORDS.copy()
IMPORTANT_TOKEN_EXCLUDE_POS = set(['PART', 'PUNCT', 'DET', 'PRON', 'SYM', 'SPACE'])
NOUN_POS = set(['NOUN', 'PROPN'])
ADJ_VERB_POS = set(['ADJ', 'VERB'])
NOUN_ADJ_VERB_POS = set(['NOUN', 'ADJ', 'VERB'])
END_OF_SENTENCE_PUNCT = {'.', '?', '!'}


class Token:
    """
    This class extracts the essential information from a SpacyToken
    """
    def __init__(self, token: st.Token, stop_words: set[str]) -> None:
        _lemma = token.lemma_.lower()
        self.text = token.text
        self.lemma = _lemma
        self.is_stop_word = token.is_stop or \
            _lemma in stop_words or \
            token.text.lower() in stop_words
        self.embeddings = token.vector
        self.part_of_speech = token.pos_
        self.is_punctuation = token.is_punct or \
            token.dep_ == 'punct' or \
            token.pos_ == 'PUNCT'
        # TODO this isn't working
        self.is_special = token.pos_ == 'SYM' or (token.pos_ == 'X' and not token.is_alpha)
        self.is_alpha = token.is_alpha
        self.is_numeric = token.is_digit
        self.is_ascii = token.is_ascii
        self.dep = token.dep_
        self.entity_type = token.ent_type_
        self.sentiment = token.sentiment

    def __str__(self) -> str:
        return self.text

    def to_dict(self) -> dict:
        return self.__dict__

    @property
    def is_important(self):
        return not self.is_stop_word and self.part_of_speech not in IMPORTANT_TOKEN_EXCLUDE_POS


def _valid_noun_phrase(token_a: Token, token_b: Token) -> bool:
    """
    This function defines the logic to determine if two tokens combine to form a valid noun-phrase.
    """
    return token_a.is_important and \
        token_b.is_important and \
        token_a.part_of_speech in NOUN_ADJ_VERB_POS and \
        token_b.part_of_speech in NOUN_ADJ_VERB_POS


class Document:
    def __init__(self, tokens: list[Token], text_original: str, text_cleaned: str):
        self._tokens = tokens
        self._text_original = text_original
        self._text_cleaned = text_cleaned

    def text(self, original=True):
        if original:
            return self._text_original
        else:
            return self._text_cleaned

    def num_important_tokens(self) -> int:
        return sum(t.is_important for t in self._tokens)

    def to_dict(self) -> dict:
        if len(self) == 0:
            return {}
        token_keys = self._tokens[0].to_dict().keys()
        _dict = {x: [None] * len(self) for x in token_keys}
        for i, token in enumerate(self):
            for k in token_keys:
                _dict[k][i] = getattr(token, k)
        return _dict

    def lemmas(self, important_only: bool = True) -> Iterable[str]:
        """
        This function returns the lemmas of the document.

        Args:
            important_only:
                if True, return the lemma if the Token's `is_important` property is True.
                if False, return the lemma if the Token's `is_important` property is True.
        """
        if important_only:
            return (t.lemma for t in self._tokens if t.is_important)
        else:
            return (
                t.lemma for t in self._tokens
                if (not t.is_punctuation and not t.is_special) or (t.text in END_OF_SENTENCE_PUNCT)
            )

    def n_grams(self, n: int = 2, separator: str = '-') -> Iterable[str]:
        _tokens = [
            t for t in self._tokens if not t.is_punctuation or t.text in END_OF_SENTENCE_PUNCT
        ]
        return (
            separator.join(t.lemma for t in ngram) for ngram in zip(*[_tokens[i:] for i in range(n)])  # noqa
            if all(t.is_important for t in ngram)
        )

    def nouns(self) -> Iterable[str]:
        return (t.lemma for t in self._tokens if t.part_of_speech in NOUN_POS)

    def noun_phrases(self, separator: str = '-') -> Iterable[str]:
        _tokens = [
            t for t in self._tokens if not t.is_punctuation or t.text in END_OF_SENTENCE_PUNCT
        ]
        return (
            separator.join(t.lemma for t in ngram) for ngram in zip(*[_tokens[i:] for i in range(2)])  # noqa
            if _valid_noun_phrase(token_a=ngram[0], token_b=ngram[1])
        )

    def adjectives_verbs(self) -> Iterable[str]:
        return (t.lemma for t in self._tokens if t.part_of_speech in ADJ_VERB_POS)

    def entities(self) -> Iterable[str]:
        return ((t.text, t.entity_type) for t in self._tokens if t.entity_type)

    def token_embeddings(self) -> np.array:
        """
        This function returns the vectors of the important tokens (i.e. not a stop word and not
        punctuation).
        """
        return np.array([t.embeddings for t in self._tokens if t.is_important])

    def embeddings(self, aggregation: str = 'average') -> np.array:
        """This function aggregates the individual token vectors into one document vector."""
        if len(self.token_embeddings()) == 0:
            return self.token_embeddings()

        if aggregation == 'average':
            return self.token_embeddings().mean(axis=0)
        else:
            raise ValueError(f"{aggregation} value not supported for `vector()`")

    @lru_cache()
    def diff(self, use_lemmas: bool = False) -> str:
        """
        Returns HTML containing diff between `text_original` and either the cleaned text or,
        optionally, the lemmas.

        Args:
            use_lemmas:
                If `True`, diff the original text of against all of the lemmas.
        """
        if use_lemmas:
            return diff_text(
                text_a=self._text_original,
                text_b=' '.join(self.lemmas(important_only=False))
            )
        else:
            return diff_text(text_a=self._text_original, text_b=self._text_cleaned)

    @lru_cache()
    def sentiment(self) -> float:
        _sentiment = [t.sentiment for t in self._tokens if t.is_important]
        if len(_sentiment) == 0:
            return 0
        return sum(_sentiment) / len(_sentiment)

    @lru_cache()
    def impurity(
            self,
            pattern: str = rp.SUSPICIOUS,
            min_length: int = 10,
            original=False) -> float:
        """
        Returns the percent of characters matching regex_patterns.SUSPICIOUS from either the
        cleaned text (i.e. `text` property) if `original` if False) or the `text_original` property
        (if `original` is `True`).

        Args:
            pattern:
                regex pattern to use to search for suspicious characters; default is
                regex_patterns.SUSPICIOUS
            min_length:
                the minimum length of `text` required to return a score; if `text` is less than the
                minimum length, then a value of np.nan is returned.
            original:
                if True use `text_original`, if False, use `text.
        Copied from:
            Blueprints for Text Analytics Using Python
            by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
            (O'Reilly, 2021), 978-1-492-07408-3.
            https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch04/Data_Preparation.ipynb
        """
        text = self._text_original if original else self._text_cleaned
        if text is None or len(text) < min_length:
            return np.nan
        else:
            return len(regex.findall(pattern, text))/len(text)

    def __str__(self) -> str:
        return self._text_original if self._text_original else self._text_cleaned

    def __len__(self):
        return len(self._tokens)

    def __iter__(self):
        for token in self._tokens:
            yield token

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._tokens[index]
        elif isinstance(index, slice):
            return self._tokens[index.start:index.stop:index.step]
        else:
            raise TypeError("Invalid index type")


class Corpus:
    def __init__(
            self,
            stop_words_to_add: Union[set[str], None] = None,
            stop_words_to_remove: Union[set[str], None] = None,
            tokenizer: Optional[stz.Tokenizer] = None,
            pre_process: Optional[Callable] = None,
            spacy_model: str = 'en_core_web_sm',
            sklearn_tokenenizer_min_df: int = 5,
            sklearn_tokenenizer_include_bi_grams: bool = True,
            sklearn_tokenenizer_max_tokens: Optional[int] = None,
            sklearn_tokenenizer_max_bi_grams: Optional[int] = None):
        """
        Args:
            stopwords_to_add: stop words to add
            stopwords_to_remove: stop words to remove
            tokenizer: a custom tokenizer
            pre_process:
                a function that takes a string as input and returns the string after cleaning/
                pre-processing (this function will be called before tokenizing.)
            spacy_model:
                the spacy model to use
                (e.g. 'en_core_web_sm', 'en_core_web_md', 'en_core_web_lg')
        """
        # spacy.load caches this language model and any stop words we have added/removed
        # calling spacy.load on subsequent calls loads in the cached model, nothing is reset
        # so we need reset the default stop words
        self.documents = None
        self.__count_vectorizer = None
        self._count_matrix = None
        self.__tf_idf_vectorizer = None
        self._tf_idf_matrix = None
        self.pre_process = pre_process
        self._nlp = spacy.load(spacy_model)
        self._nlp.Defaults.stop_words = STOP_WORDS_DEFAULT.copy()
        self._include_bi_grams = sklearn_tokenenizer_include_bi_grams
        self._max_tokens = sklearn_tokenenizer_max_tokens
        self._max_bi_grams = sklearn_tokenenizer_max_bi_grams
        self._min_df = sklearn_tokenenizer_min_df

        # https://machinelearningknowledge.ai/tutorial-for-stopwords-in-spacy/#i_Stopwords_List_in_Spacy
        if stop_words_to_add is not None:
            if isinstance(stop_words_to_add, list):
                stop_words_to_add = set(stop_words_to_add)
            self._nlp.Defaults.stop_words |= stop_words_to_add

        if stop_words_to_remove is not None:
            if isinstance(stop_words_to_remove, list):
                stop_words_to_remove = set(stop_words_to_remove)
            self._nlp.Defaults.stop_words -= stop_words_to_remove

        # this is dumb, but it appears that i have to call `spacy.load` again;
        # because it doesn't look like token.is_stop updates immediately after
        # nlp.Defaults.stop_words are updated
        # it seems as though I have to call spacy.load again
        # e.g. if I run the following code
        #     nlp = spacy.load("en_core_web_sm")
        #     print([t.is_stop for t in nlp("hello xyz")])
        #     nlp.Defaults.stop_words |= {"xyz"}
        #     print([t.is_stop for t in nlp("hello xyz")])
        # it prints
        #     [False, False]
        #     [False, False]   # i would expect this to return [False, True]
        # however if i run
        #     nlp = spacy.load("en_core_web_sm")
        #     print([t.is_stop for t in nlp("hello xyz")])
        #     nlp.Defaults.stop_words |= {"xyz"}
        #     nlp = spacy.load("en_core_web_sm")  # ***re-load
        #     print([t.is_stop for t in nlp("hello xyz")])
        # it prints
        #    [False, False]
        #    [False, True]   # returns [False, True] as expected
        self._nlp = spacy.load(spacy_model)
        if tokenizer:
            self._nlp.tokenizer = tokenizer(self._nlp)
        else:
            self._nlp.tokenizer = custom_tokenizer(self._nlp)

    def fit(self, documents: list[str]):
        # TODO: implement parallel processing logic
        _original = documents.copy()
        if self.pre_process:
            documents = [self.pre_process(x) for x in documents]
        docs = self._nlp.pipe(documents)
        documents = [None] * len(documents)
        for i, doc in enumerate(docs):
            tokens = [None] * len(doc)
            for j, token in enumerate(doc):
                tokens[j] = Token(token=token, stop_words=self.stop_words)
            documents[i] = Document(
                tokens=tokens,
                text_original=_original[i],
                text_cleaned=str(doc),
            )
        self.documents = documents

    def _text_to_doc(self, text: str) -> Document:
        text_original = text
        if self.pre_process:
            text_clean = self.pre_process(text)
        else:
            text_clean = text.strip()
        doc = self._nlp(text_clean)
        tokens = [None] * len(doc)
        for j, token in enumerate(doc):
            tokens[j] = Token(token=token, stop_words=self.stop_words)
        return Document(tokens=tokens, text_original=text_original, text_cleaned=text_clean)

    def _prepare_doc_for_vectorizer(self, document: Document) -> str:
        max_lemmas = self._max_tokens or len(document)
        lemmas = itertools.islice(document.lemmas(important_only=True), max_lemmas)
        vectorizer_text = ' '.join(lemmas)
        if self._include_bi_grams:
            max_bi_grams = self._max_bi_grams or len(document)
            bi_grams = itertools.islice(document.n_grams(n=2), max_bi_grams)
            bi_grams = ' '.join(bi_grams)
            vectorizer_text += ' ' + bi_grams

        return vectorizer_text

    @property
    def stop_words(self) -> set[str]:
        return self._nlp.Defaults.stop_words

    def text(self, original=True) -> Iterable[str]:
        return (d.text(original=original) for d in self.documents)

    def lemmas(self, important_only: bool = True) -> Iterable[str]:
        """
        This function returns the lemmas of the important tokens (i.e. not a stop word and not
        punctuation).

        Args:
            important_only: if True, for each documment return the lemmas if the Token's
            `important` property is True.
        """
        return (d.lemmas(important_only=important_only) for d in self.documents)

    def n_grams(self, n: int = 2, separator: str = '-') -> Iterable[str]:
        return (d.n_grams(n=n, separator=separator) for d in self.documents)

    def nouns(self) -> Iterable[str]:
        return (d.nouns() for d in self.documents)

    def noun_phrases(self, separator: str = '-') -> Iterable[str]:
        return (d.noun_phrases(separator=separator) for d in self.documents)

    def adjectives_verbs(self) -> Iterable[str]:
        return (d.adjectives_verbs() for d in self.documents)

    def entities(self) -> Iterable[str]:
        return (d.entities() for d in self.documents)

    def diff(self, first_n: Optional[int] = None, use_lemmas: bool = False) -> str:
        """
        Returns HTML containing diff between the original and cleaned text for each document.

        Args:
            first_n: int:
                if None, return diff for all documents; if int, return the diff for the `first_n`
                documents.
            use_lemmas:
                If `True`, diff the original text of against all of the lemmas instead of the
                cleaned text.
        """
        if not first_n:
            first_n = len(self.documents)

        if use_lemmas:
            _lemmas = itertools.islice(self.lemmas(important_only=False), first_n)
            text_b = [' '.join(x) for x in _lemmas]
        else:
            _documents = itertools.islice(self.documents, first_n)
            text_b = [x.text(original=False) for x in _documents]

        return diff_text(
            text_a=[x.text(original=True) for x in self.documents[0:first_n]],
            text_b=text_b
        )

    def _doc_to_tf_idf_embeddings(self, document: Document) -> np.array:


    @lru_cache()
    def embeddings_matrix(self, aggregation='average') -> np.array:
        """
        Returns the embeddings_matrix for the corpus, where each row of the matrix is the
        corresponding document's aggregated embeddings (from the individual Token's embeddings).
        Embeddings can be aggregated either by averaging all of the token's embeddings or weighting
        each of the token's embeddings by the TF-IDF values for that document/token.

        Args:
            aggregation: values can be `average` or `tf_idf`, as described above.
        """
        def _pad_vectors(_vectors):
            """
            Ensure vectors/embeddings are all the same size (empty documents will have empty
            embeddings).
            """
            # Find the length of the longest array in _vectors
            _max_len = max([x.size for x in _vectors])
            # Pad the shorter arrays with zeros to match the length of the longest array
            _vectors = [
                np.pad(x, (0, _max_len - x.size), mode='constant') if x.size > 0 else np.zeros(_max_len)  # noqa
                for x in _vectors
            ]
            return _vectors

        if aggregation == 'average':
            embeddings_vectors = [d.embeddings(aggregation=aggregation) for d in self.documents]
            return np.array(_pad_vectors(embeddings_vectors))

        elif aggregation == 'tf_idf':
            # we'll use idf as the weight, which has greater values for words that are less
            # frequent across all documents
            # we ignore the Term-Frequency part because that is the number of times the term
            # appears in a document, which will already be handled as we iterate through each
            # token. So if a particular token appears multiple times in a doc it will get weighted
            # multple times
            idf_lookup = dict(zip(
                self.tf_idf_vocabulary(),
                self._tf_idf_vectorizer().idf_
            ))
            embeddings_vectors = [None] * len(self)
            for i, document in enumerate(self):
                # each row in `token_embeddings` matrix corresponds to a lemma in
                # `lemmas(important_only=True)` and the lemma's embedding
                token_embeddings = document.token_embeddings()
                if len(token_embeddings) == 0:
                    embeddings_vectors[i] = np.array([])
                else:
                    assert token_embeddings.shape[0] == document.num_important_tokens()
                    lemmas = list(document.lemmas(important_only=True))
                    assert len(lemmas) == token_embeddings.shape[0]
                    # for each lemma (which corresponds to a row in the token_embeddings matrix) let's
                    # get the **IDF** weight
                    # some lemmas (e.g. _number_ or lemmas that didn't make the minimum document
                    # frequency) will not have IDF values and so we'll assign them a value/weight of 0
                    weights = np.array([idf_lookup.get(x, 0) for x in lemmas])
                    # Normalize the weights to sum to 1; when used with the dot product will get a
                    # weighted average
                    weights = weights / np.sum(weights)
                    assert len(weights) == token_embeddings.shape[0]
                    doc_weighted_embedding = weights.dot(token_embeddings)
                    assert doc_weighted_embedding.shape == (token_embeddings.shape[1],)
                    embeddings_vectors[i] = doc_weighted_embedding

            return np.array(_pad_vectors(embeddings_vectors))

        else:
            raise ValueError(f"Invalid value of `aggregation`: '{aggregation}'")

    def _count_vectorizer(self):
        if self.__count_vectorizer is None:
            vectorizer_text = [self._prepare_doc_for_vectorizer(d) for d in self.documents]
            self.__count_vectorizer = CountVectorizer(
                stop_words=None,  # already removed via _prepare_doc_for_vectorizer via
                token_pattern="(?u)\\b[\\w-]+\\b",
                min_df=self._min_df,
            )
            self._count_matrix = self.__count_vectorizer.fit_transform(vectorizer_text)

        return self.__count_vectorizer

    def text_to_count_vector(self, text):
        document = self._text_to_doc(text=text)
        vectorizer_text = self._prepare_doc_for_vectorizer(document=document)
        return self._count_vectorizer().transform([vectorizer_text])

    def count_vocabulary(self):
        return self._count_vectorizer().get_feature_names_out()

    def count_matrix(self):
        if self._count_matrix is None:
            _ = self._count_vectorizer()  # run vectorizer and initial matrix
        return self._count_matrix

    def _tf_idf_vectorizer(self):
        if self.__tf_idf_vectorizer is None:
            vectorizer_text = [self._prepare_doc_for_vectorizer(d) for d in self.documents]
            self.__tf_idf_vectorizer = TfidfVectorizer(
                stop_words=None,  # already removed via _prepare_doc_for_vectorizer
                token_pattern="(?u)\\b[\\w-]+\\b",
                min_df=self._min_df,
            )
            self._tf_idf_matrix = self.__tf_idf_vectorizer.fit_transform(vectorizer_text)

        return self.__tf_idf_vectorizer

    def text_to_tf_idf_vector(self, text):
        document = self._text_to_doc(text=text)
        vectorizer_text = self._prepare_doc_for_vectorizer(document=document)
        return self._tf_idf_vectorizer().transform([vectorizer_text])

    def tf_idf_vocabulary(self):
        return self._tf_idf_vectorizer().get_feature_names_out()

    def tf_idf_matrix(self):
        if self._tf_idf_matrix is None:
            _ = self._tf_idf_vectorizer()  # run vectorizer and initial matrix
        return self._tf_idf_matrix

    def token_count(self, groups: list) -> pd.DataFrame:
        """
        The value in index `i` in `groups` corresponds to document `i` in the corpus; groups must
        pass list of values equal to the amount of documents in teh corpus.
        """
        df = pd.DataFrame(dict(
            tokens=self.count_vocabulary(),
            count=self.count_matrix().sum(axis=0).A1,
        ))
        df.sort_values('count', ascending=False, inplace=True)
        return df

    def tf_idf(self, groups: list) -> pd.DataFrame:
        """
        The value in index `i` in `groups` corresponds to document `i` in the corpus; groups must
        pass list of values equal to the amount of documents in teh corpus.
        """
        df = pd.DataFrame(dict(
            tokens=self.tf_idf_vocabulary(),
            tf_idf=self.tf_idf_matrix().sum(axis=0).A1,
        ))
        df.sort_values('tf_idf', ascending=False, inplace=True)
        return df

    @lru_cache()
    def similarity_matrix(self, how: str):
        """
        Uses cosine_similarity.
        NOTE: the similarity matrix for a given document/row will have all values of 0 if the
        document is empty. In scikit-learn, the cosine_similarity of two vectors of all 0's is 0.
        """
        if how == 'embedding-average':
            return cosine_similarity(X=self.embeddings_matrix(aggregation='average'))
        elif how == 'embedding-tf_idf':
            return cosine_similarity(X=self.embeddings_matrix(aggregation='tf_idf'))
        elif how == 'count':
            return cosine_similarity(X=self.count_matrix())
        elif how == 'tf_idf':
            return cosine_similarity(X=self.tf_idf_matrix())
        else:
            raise ValueError("Invalid value passed to `how`")

    @lru_cache()
    def calculate_similarity(self, text: str, how: str) -> np.array:
        """
        Similarity is calculated based on cosine_similarity from scikit-learn, and because of the
        way it is calculated, two empty documents (i.e. empty vectors) will have a similarity of 0.
        As a default, doesn't seem like a bad thing.
        """
        if how == 'embedding-average':
            document = self._text_to_doc(text=text)
            return cosine_similarity(
                X=self.embeddings_matrix(aggregation='average'),
                Y=document.embeddings(aggregation='average')
            ).flatten()
        elif how == 'embedding-tf_idf':
            return cosine_similarity(X=self.embeddings_matrix(aggregation='tf_idf')).flatten()
        elif how == 'count':
            count_vector = self.text_to_count_vector(text=text)
            return cosine_similarity(X=self.count_matrix(), Y=count_vector).flatten()
        elif how == 'tf_idf':
            tf_idf_vector = self.text_to_tf_idf_vector(text=text)
            return cosine_similarity(X=self.tf_idf_matrix(), Y=tf_idf_vector).flatten()
        else:
            raise ValueError("Invalid value passed to `how`")

    @singledispatchmethod
    def get_similar_docs(self, obj: object, top_n: int):
        raise NotImplementedError("Invalid Type")

    @get_similar_docs.register
    def _(self, index: int, top_n: int = 10):
        # based on document at `index`, what are the `top_n` most similar docs
        return "pass integer"

    @get_similar_docs.register
    def _(self, text: str, top_n: int = 10):
        # based on document `doc`, what are the `top_n` most similar docs
        return "pass string"

    def plot_word_cloud():
        pass

    def get_context():
        pass
    # def __str__(self) -> str:
    #     return [t.text for t in self.tokens]

    def __len__(self):
        return len(self.documents)

    def __iter__(self):
        for document in self.documents:
            yield document

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.documents[index]
        elif isinstance(index, slice):
            return self.documents[index.start:index.stop:index.step]
        else:
            raise TypeError("Invalid index type")


# class DocumentProcessor:
    

    # def extract(
    #         self,
    #         documents: Collection[str],
    #         all_lemmas: bool = True,
    #         partial_lemmas: bool = True,
    #         bi_grams: bool = True,
    #         adjectives_verbs: bool = True,
    #         nouns: bool = True,
    #         noun_phrases: bool = True,
    #         named_entities: bool = True) -> dict[list[list[str]]]:
    #     if self.pre_process:
    #         documents = [self.pre_process(x) for x in documents]
    #     docs = self._nlp.pipe(documents)

    #     extracted_values = [None] * len(documents)
    #     for j, doc in enumerate(docs):
    #         extracted_values[j] = extract_from_doc(
    #             doc=doc,
    #             stop_words=self.stop_words,
    #             all_lemmas=all_lemmas,
    #             partial_lemmas=partial_lemmas,
    #             bi_grams=bi_grams,
    #             adjectives_verbs=adjectives_verbs,
    #             nouns=nouns,
    #             noun_phrases=noun_phrases,
    #             named_entities=named_entities,
    #         )

    #     return _list_dicts_to_dict_lists(list_of_dicts=extracted_values)

    def text_to_dict(self, text: str) -> dict[list[str]]:
        """
        This code takes text and returns a dictionary of various attributes extracted from spaCy.
        Args:
            text: the text to process
            stop_words: stop-words
        """
        doc = self._nlp(text)
        tokens = dict(
            text=[None] * len(doc),
            lemma=[None] * len(doc),
            is_stop=[None] * len(doc),
            is_punct=[None] * len(doc),
            vector=[None] * len(doc),
            is_punctuation=[None] * len(doc),
            is_special=[None] * len(doc),
            is_alpha=[None] * len(doc),
            is_numeric=[None] * len(doc),
            is_ascii=[None] * len(doc),
            dep=[None] * len(doc),
            part_of_speech=[None] * len(doc),
            entity_type=[None] * len(doc),
            sentiment=[None] * len(doc),
        )
        for i, token in enumerate(doc):
            _lemma = token.lemma_.lower()
            tokens['text'][i] = token.text
            tokens['lemma'][i] = _lemma
            tokens['is_stop'][i] = token.is_stop or \
                _lemma in self.stop_words or \
                text.lower() in self.stop_words
            tokens['is_punct'][i] = token.pos_ == 'X' and not token.is_alpha
            tokens['vector'][i] = token.vector
            tokens['is_punctuation'][i] = token.is_punct or \
                token.dep_ == 'punct' or \
                token.pos_ == 'PUNCT'
            tokens['is_special'][i] = token.pos_ == 'SYM' or (token.pos_ == 'X' and not token.is_alpha)
            tokens['is_alpha'][i] = token.is_alpha
            tokens['is_numeric'][i] = token.is_numeric
            tokens['is_ascii'][i] = token.is_ascii
            tokens['dep'][i] = token.dep_
            tokens['part_of_speech'][i] = token.pos_
            tokens['entity_type'][i] = token.ent_type_
            tokens['sentiment'][i] = token.sentiment
        return tokens

    # def text_to_dataframe(self, text: str, include_punctuation: bool = False) -> pd.DataFrame:
    #     """
    #     This code takes a spaCy Doc and converts the doc into a pd.DataFrame

    #     This code is modified from:
    #         Blueprints for Text Analytics Using Python
    #         by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
    #         (O'Reilly, 2021), 978-1-492-07408-3.
    #         https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch04/Data_Preparation.ipynb

    #     Args:
    #         doc: the doc to convert
    #         include_punctuation: whether or not to return the punctuation in the DataFrame.
    #     """
    #     doc = self._nlp(text)
    #     rows = []
    #     for i, t in enumerate(doc):
    #         if (not t.is_punct and t.pos_ != 'PUNCT') or include_punctuation:
    #             row = {
    #                 'token': i, 'text': t.text, 'lemma_': t.lemma_,
    #                 'is_stop': t.is_stop, 'is_alpha': t.is_alpha,
    #                 'pos_': t.pos_, 'dep_': t.dep_,
    #                 'ent_type_': t.ent_type_, 'ent_iob_': t.ent_iob_
    #             }
    #             rows.append(row)

    #     df = pd.DataFrame(rows).set_index('token')
    #     df.index.name = None
    #     return df


def _list_dicts_to_dict_lists(list_of_dicts: list[dict]):
    """
    Args:
        list_of_dicts:
            e.g. `[{'x': 1, 'y': 10}, {'x': 2, 'y': 11}, {'x': 3, 'y': 12}]`
    Returns:
        e.g. `{'x': [1, 2, 3], 'y': [10, 11, 12]}`
    """
    dict_of_lists = {}
    for dictionary in list_of_dicts:
        for key, value in dictionary.items():
            dict_of_lists.setdefault(key, []).append(value)

    return dict_of_lists


list_of_dicts = [{'x': 1, 'y': 10}, {'x': 2, 'y': 11}, {'x': 3, 'y': 12}]
expected = {'x': [1, 2, 3], 'y': [10, 11, 12]}
assert _list_dicts_to_dict_lists(list_of_dicts) == expected

list_of_dicts = [{'y': 10}, {'x': 2, 'y': 11}, {'x': 3}, {'x': 4}]
expected = {'x': [2, 3, 4], 'y': [10, 11]}
assert _list_dicts_to_dict_lists(list_of_dicts) == expected


def custom_tokenizer(nlp: Language) -> stz.Tokenizer:
    """
    This code creates a custom tokenizer, as described on pg. 108 of

        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
        https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch04/Data_Preparation.ipynb

    Args:
        nlp: the Language
    """
    prefixes = [
        pattern for pattern in nlp.Defaults.prefixes if pattern not in ['-', '_', '#']
    ]
    suffixes = [
        pattern for pattern in nlp.Defaults.suffixes if pattern not in ['_']
    ]
    infixes = nlp.Defaults.infixes
    #     [
    #     pattern for pattern in nlp.Defaults.infixes if not re.search(pattern, 'xx-xx')
    # ]

    return stz.Tokenizer(
        vocab=nlp.vocab,
        rules=nlp.Defaults.tokenizer_exceptions,
        prefix_search=compile_prefix_regex(prefixes).search,
        suffix_search=compile_suffix_regex(suffixes).search,
        infix_finditer=compile_infix_regex(infixes).finditer,
        token_match=nlp.Defaults.token_match
    )


def extract_lemmas(doc: spacy.tokens.doc.Doc,
                   exclude_stopwords: bool = True,
                   stop_words: Optional[set] = None,
                   exclude_punctuation: bool = True,
                   exclude_numbers: bool = False,
                   include_part_of_speech: Union[List[str], None] = None,
                   exclude_part_of_speech: Union[List[str], None] = None,
                   min_frequency: int = 1) -> list[str]:
    """
    This function extracts lemmas from the `doc`.

    This code is modified from:
        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
        https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch04/Data_Preparation.ipynb

    Args:
        doc: the doc to extract from
        exclude_stopwords: if True, exclude stopwords
        stop_words:
            list of stop-words; needed to remove lemmas that are stop words, spacy does not seem
            to do this.
        exclude_punctuation: if True, exclude punctuation
        exclude_numbers: if True, exclude numbers
        include_part_of_speech:  e.g. ['ADJ', 'NOUN']
        exclude_part_of_speech: e.g. ['ADJ', 'NOUN']
        min_frequency: the minimum frequency required for the lemma
    """
    words = textacy.extract.words(  # noqa
        doc,
        filter_stops=exclude_stopwords,
        filter_punct=exclude_punctuation,
        filter_nums=exclude_numbers,
        include_pos=include_part_of_speech,
        exclude_pos=exclude_part_of_speech,
        min_freq=min_frequency,
    )
    temp = next(words)
    temp.vector.shape
    tokens = [token if (token := t.lemma_.lower()) != 'datum' else 'data' for t in words]
    if exclude_stopwords and stop_words:
        # exclude_stopwords appears to remove pre-lemmatized stop-words
        return [t for t in tokens if t not in stop_words]

    return tokens


def extract_n_grams(doc: spacy.tokens.doc.Doc,
                    n=2,
                    sep: str = ' ',
                    exclude_stopwords: bool = True,
                    stop_words: Optional[set] = None,
                    exclude_punctuation: bool = True,
                    exclude_numbers: bool = False,
                    include_part_of_speech: Union[List[str], None] = None,
                    exclude_part_of_speech: Union[List[str], None] = None,
                    ):
    """
    This function extracts n_grams from the `doc`.

    Note that exclude_punctuation and exclude_numbers does not seem to work; punctuation and
    numbers are being returned.


    Args:
        doc: the doc to extract from
        n: the number of grams to return
        sep: the string that will separate the n grams.
        exclude_stopwords: if True, exclude stopwords
        stop_words:
            list of stop-words; needed to remove lemmas that are stop words, spacy does not seem
            to do this.
        exclude_punctuation: if True, exclude punctuation
        exclude_numbers: if True, exclude numbers
        include_part_of_speech:  e.g. ['ADJ', 'NOUN']
        exclude_part_of_speech: e.g. ['ADJ', 'NOUN']
    """
    spans = textacy.extract.basics.ngrams(  # noqa
        doc,
        n=n,
        filter_stops=exclude_stopwords,
        filter_punct=exclude_punctuation,
        filter_nums=exclude_numbers,
        include_pos=include_part_of_speech,
        exclude_pos=exclude_part_of_speech,
    )
    spans = list(spans)
    n_grams = [
        sep.join([token if (token := t.lemma_.lower()) != 'datum' else 'data' for t in s])
        for s in spans
    ]
    if exclude_stopwords and stop_words:
        has_stopwords = [any([t.lemma_.lower() in stop_words for t in s]) for s in spans]
        return [gram for stop, gram in zip(has_stopwords, n_grams) if not stop]

    return n_grams


def extract_noun_phrases(doc: spacy.tokens.doc.Doc,
                         exclude_stopwords: bool = True,
                         stop_words: Optional[set] = None,
                         preceding_part_of_speech: Union[List[str], None] = None,
                         subsequent_part_of_speech: Union[List[str], None] = None,
                         sep: str = ' '):
    """
    This function extracts the "noun phrases" from `doc` and returns the lemmas.

    This code is modified from:
        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
        https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch04/Data_Preparation.ipynb

    Args:
        doc:
            the doc to extract from
        exclude_stopwords:
            if True, exclude stopwords
        stop_words:
            list of stop-words; needed to remove lemmas that are stop words, spacy does not seem
            to do this.
        preceding_part_of_speech:
            Part of Speech to filter for in the preceding word. If None, default is ['NOUN', 'ADJ',
            'VERB']
        subsequent_part_of_speech:
            Part of Speech to filter for in the subsequent word. If None, default is ['NOUN',
            'ADJ', 'VERB']
        sep:
            the separator to join the lemmas on.
    """
    if preceding_part_of_speech is None:
        preceding_part_of_speech = ['NOUN', 'ADJ', 'VERB']
    if subsequent_part_of_speech is None:
        subsequent_part_of_speech = ['NOUN', 'ADJ', 'VERB']

    patterns = []
    for pos in preceding_part_of_speech:
        patterns.append(f"POS:{pos} POS:NOUN:+")
    for pos in subsequent_part_of_speech:
        patterns.append(f"POS:NOUN POS:{pos}:+")

    spans = textacy.extract.matches.token_matches(
        doc,
        patterns=patterns,
    )
    spans = list(spans)
    phrases = [
        sep.join([token if (token := t.lemma_.lower()) != 'datum' else 'data' for t in s])
        for s in spans
    ]
    if exclude_stopwords and stop_words:
        # Spacy doesn't remove stopwords when the lemma itself is a stopword so we have to manually
        # remove
        has_stopwords = [any([t.lemma_.lower() in stop_words for t in s]) for s in spans]
        return [phrase for stop, phrase in zip(has_stopwords, phrases) if not stop]

    return phrases


def extract_named_entities(doc: spacy.tokens.doc.Doc,
                           include_types: Union[List[str], None] = None,
                           sep: str = ' ',
                           include_label: bool = True):
    """
    This function extracts the named entities from the doc.

    This code is modified from:
        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
        https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch04/Data_Preparation.ipynb

    Args:
        doc: the doc to extract from
        include_types:  Part of Speech to filter to include.
        sep: the separator to join the lemmas on.
        include_label: if True, include the type (i.e. label) of the named entity.
    """
    entities = textacy.extract.entities(  # noqa
        doc,
        include_types=include_types,
        exclude_types=None,
        drop_determiners=True,
        min_freq=1
    )

    def format_named_entity(entity):
        lemmas = sep.join([t.lemma_ for t in entity])

        value = lemmas
        if include_label:
            value = f'{value} ({entity.label_})'
        return value

    return [format_named_entity(e) for e in entities]


def extract_from_doc(doc: spacy.tokens.doc.Doc,
                     stop_words: Optional[set] = None,
                     all_lemmas: bool = True,
                     partial_lemmas: bool = True,
                     bi_grams: bool = True,
                     adjectives_verbs: bool = True,
                     nouns: bool = True,
                     noun_phrases: bool = True,
                     named_entities: bool = True) -> dict:
    """
    This function extracts common types of tokens from a spaCy doc.

    This code is modified from:
        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
        https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch04/Data_Preparation.ipynb

    Args:
        doc: the doc to extract from
        stop_words:
            list of stop-words; needed to remove lemmas that are stop words, spacy does not seem
            to do this.
        all_lemmas: if True, return all lemmas (lower-case) of every word, also includes
        punctuation;
            This is useful if you need to use e.g. scikit-learn TfidfVectorizer and need the raw
            document. i.e. you can do a `' '.join()` with this data.
        partial_lemmas: if True, return lemmas (lower-case), excluding stopwords and punctuation
        bi_grams: if True, return bi_grams
        adjectives_verbs: if True, return adjectives_verbs
        nouns: if True, return nouns
        noun_phrases: if True, return noun_phrases
        named_entities: if True, return named_entities
    """

    words = textacy.extract.words(  # noqa
        doc,
        filter_stops=False,
        filter_punct=True,
        filter_nums=False,
        # include_pos=include_part_of_speech,
        # exclude_pos=exclude_part_of_speech,
        min_freq=1,
    )
    results = dict()

    if all_lemmas:
        results['all_lemmas'] = extract_lemmas(
            doc,
            stop_words=None,
            exclude_stopwords=False,
            exclude_punctuation=True,
            exclude_numbers=False,
            include_part_of_speech=None,
            exclude_part_of_speech=None,
            min_frequency=1
        )
    if partial_lemmas:
        results['partial_lemmas'] = extract_lemmas(
            doc,
            exclude_part_of_speech=['PART', 'PUNCT', 'DET', 'PRON', 'SYM', 'SPACE'],
            exclude_stopwords=True,
            stop_words=stop_words,
        )
    if bi_grams:
        results['bi_grams'] = extract_n_grams(
            doc,
            n=2,
            sep='-',
            exclude_stopwords=True,
            stop_words=stop_words,
        )
    if adjectives_verbs:
        results['adjs_verbs'] = extract_lemmas(
            doc,
            include_part_of_speech=['ADJ', 'VERB'],
            exclude_stopwords=True,
            stop_words=stop_words,
        )
    if nouns:
        results['nouns'] = extract_lemmas(
            doc,
            include_part_of_speech=['NOUN', 'PROPN'],
            exclude_stopwords=True,
            stop_words=stop_words,
        )
    if noun_phrases:
        # Setting exclude_stopwords to true is extemely slow
        # need to figure this out if we decide to use noun-phrases
        results['noun_phrases'] = extract_noun_phrases(
            doc,
            exclude_stopwords=True,
            stop_words=stop_words,
            sep='-'
        )
    if named_entities:
        results['entities'] = extract_named_entities(doc)

    return results
