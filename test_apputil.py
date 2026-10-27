"""Check Markov transitions, generation, and input edge cases."""

import unittest
from unittest.mock import patch

from apputil import MarkovText


class TestMarkovText(unittest.TestCase):
    def test_transition_dictionary(self):
        model = MarkovText("a b a b a c")
        expected = {"a": ["b", "b", "c"], "b": ["a", "a"], "c": []}
        self.assertEqual(model.term_dict, expected)
        self.assertEqual(model.get_term_dict(), expected)

    def test_generation_uses_frequency_list(self):
        model = MarkovText("a b a b a c")
        with patch("apputil.random.choice", return_value="c") as choice:
            self.assertEqual(model.generate("a", 2), "a c")
        choice.assert_called_once_with(["b", "b", "c"])

    def test_seed_length_and_observed_transitions(self):
        model = MarkovText("one two one three one")
        terms = model.generate("one", 30).split()
        self.assertEqual(terms[0], "one")
        self.assertEqual(len(terms), 30)
        for current, following in zip(terms, terms[1:]):
            self.assertIn(following, model.term_dict[current])
        self.assertEqual(len(model.generate().split()), 15)

    def test_terminal_word_restarts(self):
        model = MarkovText("start end")
        with patch(
            "apputil.random.choice", side_effect=["start", "end"]
        ) as choice:
            self.assertEqual(model.generate("end", 3), "end start end")
        self.assertEqual(choice.call_args_list[0].args[0], ["start", "end"])
        self.assertEqual(choice.call_args_list[1].args[0], ["end"])

    def test_single_word_and_whitespace(self):
        model = MarkovText("  Hello!\n\tHello!  ")
        self.assertEqual(model.term_dict, {"Hello!": ["Hello!"]})
        self.assertEqual(MarkovText("alone").generate(term_count=3),
                         "alone alone alone")

    def test_invalid_seed(self):
        with self.assertRaises(ValueError):
            MarkovText("a b").generate("missing")

    def test_empty_corpus_and_term_counts(self):
        model = MarkovText("a b")
        self.assertEqual(model.generate(term_count=0), "")
        self.assertEqual(model.generate("b", 1), "b")
        self.assertEqual(MarkovText(" \n\t").term_dict, {})
        with self.assertRaises(ValueError):
            MarkovText("").generate()
        for term_count in [-1, 1.5, True]:
            with self.subTest(term_count=term_count):
                with self.assertRaises(ValueError):
                    model.generate(term_count=term_count)


if __name__ == "__main__":
    unittest.main()
