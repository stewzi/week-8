"""Generate text using a one-word Markov chain."""

from collections import defaultdict
import random


class MarkovText:
    """Learn word transitions from a whitespace-tokenized corpus."""

    def __init__(self, corpus):
        if not isinstance(corpus, str):
            raise TypeError("corpus must be a string.")
        self.corpus = corpus
        self.term_dict = None
        self.get_term_dict()

    def get_term_dict(self):
        """Store and return successors, retaining duplicates as frequencies."""
        terms = self.corpus.split()
        term_dict = defaultdict(list, {term: [] for term in terms})

        for current, following in zip(terms, terms[1:]):
            term_dict[current].append(following)

        self.term_dict = dict(term_dict)
        return self.term_dict

    def generate(self, seed_term=None, term_count=15):
        """Return term_count words, including the optional starting word.

        At a word with no observed successor, restart at a random corpus word.
        Capitalization and punctuation are preserved from the input corpus.
        """
        if (
            not isinstance(term_count, int)
            or isinstance(term_count, bool)
            or term_count < 0
        ):
            raise ValueError("term_count must be a nonnegative integer.")
        if seed_term is not None and seed_term not in self.term_dict:
            raise ValueError("seed_term must be a token in the corpus.")
        if term_count == 0:
            return ""
        if not self.term_dict:
            raise ValueError("Cannot generate text from an empty corpus.")

        terms = list(self.term_dict)
        current = seed_term if seed_term is not None else random.choice(terms)
        generated = [current]

        for _ in range(term_count - 1):
            following = self.term_dict[current]
            current = random.choice(following if following else terms)
            generated.append(current)

        return " ".join(generated)
