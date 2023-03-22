from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache, singledispatchmethod
from itertools import islice
from typing import Callable, Iterable, Optional, Union

import pandas as pd
import numpy as np
import regex
import spacy
from spacy.language import Language
import spacy.lang.en.stop_words as sw
import spacy.tokenizer as stz
import spacy.tokens as st
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from helpsk.diff import diff_text
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

    def num_tokens(self, important_only=True) -> int:
        """
        This number correpsonds to the number of tokens.

        Args:
            important_only:
                if `True`: returns the number of Tokens with `is_important` being `True`.
                Note: this will correspond to the number of items returned by
                `lemmas(important_only=True)`.
                
                if `False`: returns the number of Tokens that are not punctuation and are not 
                special characters.
                Note: this will **not** correspond to the number of items returned by
                `lemmas(important_only=False)` because those items will also include punctuation
                corresponding to the end of a sentence.
        """
        if important_only:
            return sum(t.is_important for t in self._tokens)
        else:
            return sum(
                1 for t in self._tokens
                if not t.is_punctuation and not t.is_special
            )

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
                if False, return the lemma if the token is not punctuation or a special character
                (except if the token punctuation marking the end of a sentence). The reason behind
                this is to make it easier to separate sentences via `' '.join(...)` or similar.
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


def _process_document_batch(
        documents: list[str],
        nlp_pipe: Callable,
        stop_words: set,
        pre_process: Callable) -> list[Document]:
    """
    Takes a set of "documents" (a list of strings) and returns a list of document objects.
    """
    _originals = documents.copy()
    if pre_process:
        documents = [pre_process(x) for x in documents]
    docs = nlp_pipe(documents)
    documents = [None] * len(documents)
    for i, doc in enumerate(docs):
        tokens = [None] * len(doc)
        for j, token in enumerate(doc):
            tokens[j] = Token(token=token, stop_words=stop_words)
        documents[i] = Document(
            tokens=tokens,
            text_original=_originals[i],
            text_cleaned=str(doc),
        )
    return documents


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

    def fit(self, documents: list[str], num_batches: int = 10):
        """
        Takes a list of strings and coverts that list into a list of Document objects.

        Args:
            documents: list of strings/text
            num_batches:
                The number of batches for parallel processing. Turn off parallel processing by
                passing 0 or None.
        """
        if not num_batches or len(documents) < num_batches * 2:
            self.documents = _process_document_batch(
                documents=documents,
                nlp_pipe=self._nlp.pipe,
                stop_words=self.stop_words,
                pre_process=self.pre_process
            )
        else:
            def split_list(items, n):
                """Split a list into n sublists of equal size."""
                k, m = divmod(len(items), n)
                return list(items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

            batches = split_list(items=documents, n=num_batches)
            assert len(batches) == num_batches
            with ProcessPoolExecutor() as pool:
                pipes = [self._nlp.pipe] * num_batches
                stop_words = [self.stop_words] * num_batches
                pre_processes = [self.pre_process] * num_batches
                results = list(pool.map(
                    _process_document_batch,
                    batches,
                    pipes,
                    stop_words,
                    pre_processes
                ))
            # results will be a list of list of Documents; we want to flatten the list
            self.documents = [item for sublist in results for item in sublist]
            assert len(self.documents) == len(documents)

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
        lemmas = islice(document.lemmas(important_only=True), max_lemmas)
        vectorizer_text = ' '.join(lemmas)
        if self._include_bi_grams:
            max_bi_grams = self._max_bi_grams or len(document)
            bi_grams = islice(document.n_grams(n=2), max_bi_grams)
            bi_grams = ' '.join(bi_grams)
            vectorizer_text += ' ' + bi_grams

        return vectorizer_text

    @property
    def stop_words(self) -> set[str]:
        return self._nlp.Defaults.stop_words

    def text(self, original=True) -> Iterable[str]:
        return (d.text(original=original) for d in self.documents)

    def lemmas(self, important_only: bool = True) -> Iterable[list]:
        """
        This function returns the lemmas of the important tokens (i.e. not a stop word and not
        punctuation).

        Args:
            important_only: if True, for each documment return the lemmas if the Token's
            `important` property is True.
        """
        return (list(d.lemmas(important_only=important_only)) for d in self.documents)

    def n_grams(self, n: int = 2, separator: str = '-') -> Iterable[list]:
        return (list(d.n_grams(n=n, separator=separator)) for d in self.documents)

    def nouns(self) -> Iterable[list]:
        return (list(d.nouns()) for d in self.documents)

    def noun_phrases(self, separator: str = '-') -> Iterable[list]:
        return (list(d.noun_phrases(separator=separator)) for d in self.documents)

    def adjectives_verbs(self) -> Iterable[list]:
        return (list(d.adjectives_verbs()) for d in self.documents)

    def entities(self) -> Iterable[list]:
        return (list(d.entities()) for d in self.documents)

    def sentiments(self) -> Iterable[float]:
        return (x.sentiment() for x in self)

    def impurities(
            self,
            pattern: str = rp.SUSPICIOUS,
            min_length: int = 10,
            original=False) -> Iterable[float]:
        return (
            x.impurity(pattern=pattern, min_length=min_length, original=original) for x in self
        )

    def text_lengths(self, original=True) -> Iterable[int]:
        return (len(x.text(original=original)) for x in self)

    def num_tokens(self, important_only=True) -> Iterable[float]:
        return (x.num_tokens(important_only=important_only) for x in self)

    def to_dataframe(
            self,
            columns: Optional[str] = None,
            first_n: Optional[int] = None) -> pd.DataFrame:
        """
        Returns a DataFrame representation of the Corpus.

        Args:
            columns:
                See code for valid column names.
                use the value `all` to return all columns.
            first_n:
                only return the `first_n` documents.
        """
        if not first_n:
            first_n = len(self)

        valid_columns = [
            'text_original',
            'text_clean',
            'lemmas_all',
            'lemmas_important',
            'bi_grams',
            'nouns',
            'noun_phrases',
            'adjective_verbs',
            'sentiment',
            'impurity_original',
            'impurity_clean',
            'text_clean_length',
            'text_original_length',
            'num_tokens_all',
            'num_tokens_important_only',
        ]
        if not columns:
            columns = [
                'text_original',
                'text_clean',
                'lemmas_important',
                'bi_grams',
            ]
        elif columns == 'all':
            columns = valid_columns

        assert set(columns) <= set(valid_columns)
        df = pd.DataFrame()
        if 'text_original' in columns:
            df['text_original'] = list(islice(self.text(original=True), first_n))
        if 'text_clean' in columns:
            df['text_clean'] = list(islice(self.text(original=False), first_n))
        if 'lemmas_all' in columns:
            df['lemmas_all'] = list(islice(self.lemmas(important_only=False), first_n))
        if 'lemmas_important' in columns:
            df['lemmas_important'] = list(islice(self.lemmas(important_only=True), first_n))
        if 'bi_grams' in columns:
            df['bi_grams'] = list(islice(self.n_grams(n=2), first_n))
        if 'nouns' in columns:
            df['nouns'] = list(islice(self.nouns(), first_n))
        if 'noun_phrases' in columns:
            df['noun_phrases'] = list(islice(self.noun_phrases(), first_n))
        if 'adjective_verbs' in columns:
            df['adjective_verbs'] = list(islice(self.adjectives_verbs(), first_n))
        if 'sentiment' in columns:
            df['sentiment'] = list(islice(self.sentiments(), first_n))
        if 'impurity_original' in columns:
            df['impurity_original'] = list(islice(self.impurities(original=True), first_n))
        if 'impurity_clean' in columns:
            df['impurity_clean'] = list(islice(self.impurities(original=False), first_n))
        if 'text_original_length' in columns:
            df['text_original_length'] = list(islice(self.text_lengths(original=True), first_n))
        if 'text_clean_length' in columns:
            df['text_clean_length'] = list(islice(self.text_lengths(original=False), first_n))
        if 'num_tokens_all' in columns:
            df['num_tokens_all'] = list(islice(self.num_tokens(important_only=False), first_n))
        if 'num_tokens_important_only' in columns:
            df['num_tokens_important_only'] = list(islice(self.num_tokens(important_only=True), first_n))  # noqa
        return df[columns]

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
            _lemmas = islice(self.lemmas(important_only=False), first_n)
            text_b = [' '.join(x) for x in _lemmas]
        else:
            _documents = islice(self.documents, first_n)
            text_b = [x.text(original=False) for x in _documents]

        return diff_text(
            text_a=[x.text(original=True) for x in self.documents[0:first_n]],
            text_b=text_b
        )

    def _doc_to_tf_idf_embeddings(self, document: Document) -> np.array:
        """
        This function takes a document and returns a vector of aggregated token embeddings based
        weighted by the tf_idf matrix.
        """
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
        # each row in `token_embeddings` matrix corresponds to a lemma in
        # `lemmas(important_only=True)` and the lemma's embedding
        token_embeddings = document.token_embeddings()
        if len(token_embeddings) == 0:
            return np.array([])

        assert token_embeddings.shape[0] == document.num_tokens()
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
        return doc_weighted_embedding

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
            embeddings_vectors = [None] * len(self)
            for i, document in enumerate(self):
                embeddings_vectors[i] = self._doc_to_tf_idf_embeddings(document=document)

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
        if how == 'embeddings-average':
            return cosine_similarity(X=self.embeddings_matrix(aggregation='average'))
        elif how == 'embeddings-tf_idf':
            return cosine_similarity(X=self.embeddings_matrix(aggregation='tf_idf'))
        elif how == 'count':
            return cosine_similarity(X=self.count_matrix())
        elif how == 'tf_idf':
            return cosine_similarity(X=self.tf_idf_matrix())
        else:
            raise ValueError("Invalid value passed to `how`")

    @lru_cache()
    def calculate_similarities(self, text: str, how: str) -> np.array:
        """
        Return similarities for documents in corpus against `text`.

        Similarity is calculated based on cosine_similarity from scikit-learn, and because of the
        way it is calculated, two empty documents (i.e. empty vectors) will have a similarity of 0.
        As a default, doesn't seem like a bad thing.
        """
        if how == 'embeddings-average':
            document = self._text_to_doc(text=text)
            document_embeddings = document.embeddings(aggregation='average')
            if len(document_embeddings) == 0:
                return np.zeros(len(self))
            return cosine_similarity(
                X=self.embeddings_matrix(aggregation='average'),
                Y=document.embeddings(aggregation='average').reshape(1, -1)
            ).flatten()
        elif how == 'embeddings-tf_idf':
            document = self._text_to_doc(text=text)
            document_embeddings = self._doc_to_tf_idf_embeddings(document=document)
            if len(document_embeddings) == 0:
                return np.zeros(len(self))
            return cosine_similarity(
                X=self.embeddings_matrix(aggregation='tf_idf'),
                Y=document_embeddings.reshape(1, -1)
            ).flatten()
        elif how == 'count':
            count_vector = self.text_to_count_vector(text=text)
            return cosine_similarity(X=self.count_matrix(), Y=count_vector).flatten()
        elif how == 'tf_idf':
            tf_idf_vector = self.text_to_tf_idf_vector(text=text)
            return cosine_similarity(X=self.tf_idf_matrix(), Y=tf_idf_vector).flatten()
        else:
            raise ValueError("Invalid value passed to `how`")

    @singledispatchmethod
    def get_similar_doc_indexes(self, arg: object, how: str, top_n: int = 10) -> tuple:
        """
        Returns a tuple of two np.arrays.
        The first array is the index of documents in order of most similar to least similar.
        The second array is the corresponding similarity score for each index.
        """
        raise NotImplementedError("Invalid Type")

    @get_similar_doc_indexes.register
    def _(self, index: int, how: str, top_n: int = 10) -> tuple:
        """based on document at `index`, what are the `top_n` most similar docs"""
        # get similarities from document at `index`
        similarities = self.similarity_matrix(how=how)[index]
        index_ranks = np.argsort(-similarities)
        # filter out current index
        index_ranks = index_ranks[(index_ranks != index)]
        # only index_ranks associated with similarities above 0
        index_ranks = index_ranks[similarities[index_ranks].round(5) > 0]
        assert len(index_ranks) == len(similarities[index_ranks])
        return index_ranks[0:top_n], similarities[index_ranks][0:top_n]

    @get_similar_doc_indexes.register
    def _(self, text: str, how: str, top_n: int = 10) -> tuple:
        # based on document `doc`, what are the `top_n` most similar docs
        similarities = self.calculate_similarities(text=text, how=how)
        index_ranks = np.argsort(-similarities)
        # only index_ranks associated with similarities above 0
        index_ranks = index_ranks[similarities[index_ranks].round(5) > 0]
        assert len(index_ranks) == len(similarities[index_ranks])
        return index_ranks[0:top_n], similarities[index_ranks][0:top_n]

    def plot_word_cloud():
        pass

    def find_context():
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
