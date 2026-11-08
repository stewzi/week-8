# Week 8: Metacognitive Monitoring and Reflection

## My learning goal

My goal this week is to explain how text becomes data and how a simple Markov model generates new sequences from observed word pairs. I also want to distinguish preprocessing, vectorization, topic modeling, and sentiment analysis. My practical evidence is the completed `MarkovText` class, its checks, and the executed examples in the exercise notebook. The broader NLP methods below are concepts reviewed from the lab, rather than models I independently fitted and compared.

## Understanding the Markov property

In the exercise, each token is a state. The next token is sampled using only the current token, rather than the complete sentence history. The transition dictionary stays fixed while the generator samples from it.

**What makes this click:** The notebook's weather example separates a state from a probability distribution. If today is sunny, the example gives tomorrow probabilities of `0.9` for sunny and `0.1` for rainy. Multiplying by the transition matrix again gives the day after tomorrow probabilities of `0.86` and `0.14`. These are probabilities, not guaranteed weather predictions. Sampling a next word has the same distinction: a likely word does not have to appear on every run.

**Connection to earlier weeks:** [Week-2's NumPy work](../week-2/week2.md) helps me follow arrays and matrix operations. [Week-7's networks](../week-7/week7.md) provide another useful picture: words can be nodes, and observed word pairs can be directed edges. Repeated pairs determine how frequently an edge is sampled.

## Building and checking the transition dictionary

The implemented workflow is:

```text
Quote text -> extract quotes -> normalize whitespace -> split into tokens
          -> collect observed successors -> sample successive words
```

For the small corpus `a b a b a c`, the dictionary is:

```python
{
    "a": ["b", "b", "c"],
    "b": ["a", "a"],
    "c": [],
}
```

**How I check understanding:** I can trace each adjacent pair by hand and compare it with `get_term_dict()`. The final `c` is still a key, even though it has no observed successor. This small example is easier to verify than a large dictionary from the quote corpus.

**Misconception corrected:** Duplicate successors are useful information. Sampling uniformly from `["b", "b", "c"]` gives `b` probability `2/3` and `c` probability `1/3`. Removing duplicates would change both probabilities to `1/2`. The [Python documentation for `random.choice`](https://docs.python.org/3/library/random.html#random.choice) clarifies that it selects an element from the supplied sequence. The repeated list entries provide the frequency weighting.

Using `defaultdict(list)` also connects to the list and dictionary operations introduced earlier in the course. Its behavior is explained in the [Python collections documentation](https://docs.python.org/3/library/collections.html#collections.defaultdict). Rebuilding the dictionary starts from scratch, so repeated calls do not accidentally accumulate extra transitions.

## Generation decisions and limitations

`generate()` returns the requested number of tokens, including the starting token. A supplied `seed_term` must match a corpus token exactly; an unknown seed raises `ValueError`. Without a seed term, the generator selects a random unique corpus token.

**Important distinction:** A seed term chooses the first word. A random-number seed controls reproducibility of random sampling. Specifying the first word alone does not guarantee the same generated sentence on every run.

If the current token has an empty successor list, the implementation restarts from a random corpus token. This reaches the requested output length without sampling an empty list.

**Monitoring my assumptions:** That restart is a modeling choice, not a word pair observed in the text. Stopping at the terminal token would be another reasonable design, but could return fewer words. Joining all quotes into one corpus also creates transitions between quote boundaries. I need to remember these choices when judging the generated output.

The model preserves capitalization and punctuation through whitespace tokenization. Consequently, `kind` and `kind,` are different tokens. A sentence may look plausible locally but lose meaning over a longer passage because this model only uses one word of context. Generating text does not demonstrate understanding of its meaning.

## Preprocessing and vectorization

The [week-8 lab](lab-nlp.ipynb) introduces spaCy tokenization, lemmas, part-of-speech tags, named entities, and filtering. The [spaCy linguistic-features guide](https://spacy.io/usage/linguistic-features) helps distinguish splitting text into tokens from adding linguistic annotations.

**What helps me reason about preprocessing:** Removing frequent words may help compare document topics, but removing a negation such as `not` can change the meaning. Likewise, removing connecting words from a generation corpus can reduce readability. Cleaning decisions should follow the task rather than applying the same filter everywhere. This extends [week-3's lesson](../week-3/week3.md) that cleaning data requires preserving its meaning.

| Representation | What it captures | My understanding check |
| --- | --- | --- |
| Unigram counts | Token frequencies in each document | Can I identify the document rows and token columns? |
| TF-IDF | Term frequency adjusted by document frequency across the corpus | Can I explain why a frequent term is not always distinctive? |
| Bigrams | Adjacent two-token features | Can I explain what ordering information is retained? |
| Markov successor lists | Observed next-token choices for each current token | Can I reconstruct the word pairs by hand? |

The [scikit-learn feature-extraction guide](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) provides a reference for counts, TF-IDF, and n-grams. A unigram bag of words loses sequence information. That is an important difference from the ordered pairs used by this week's generator.

## Topic modeling, sentiment, and visualization

The lab surveys SVD/LSA, NMF, LDA, and anchored CorEx. They offer different ways to find patterns among documents and terms. The [scikit-learn decomposition guide](https://scikit-learn.org/stable/modules/decomposition.html) is a reference for the matrix-based methods and LDA.

**Question still open:** How should I choose the number of topics and judge whether they are useful? Inspecting high-weight terms alongside representative documents seems more informative than assigning a label from a few words alone. I still need hands-on practice comparing these methods.

Topic modeling concerns what documents discuss. Sentiment analysis concerns expressed tone. The lab introduces [VADER](https://github.com/cjhutto/vaderSentiment), a lexicon- and rule-based tool. A sentiment score is an estimate that needs interpretation, especially when context or sarcasm changes the apparent meaning.

**Connection to week-5:** The lab's document-term heatmaps extend [earlier visualization work](../week-5/week5.md). Before interpreting a color, I should check whether it represents a count, a TF-IDF value, or another score. A clear-looking chart does not establish that its preprocessing or interpretation is appropriate.

## Evidence and confidence checks

The executed [exercise notebook](exercises.ipynb) includes a transition-dictionary preview, generated text, a 20-token seeded example, and the output from [test_apputil.py](test_apputil.py). All seven tests passed. They cover dictionary contents, duplicate successor sampling, generation length and valid transitions, terminal restarts, whitespace and single-token inputs, unknown seeds, and empty or invalid count cases.

**How this changes my approach:** [Week-7's testing work](../week-7/week7.md) helps me check behavior rather than judge correctness from one convincing sentence. The tests inspect the choices supplied to the sampler and verify output properties. They do not prove literary quality or that a small random sample exactly matches theoretical probabilities.

## My next learning checks

- Trace a second small corpus manually before running the class.
- Compare output across several starting words and distinguish word choice from random-number seeding.
- Examine the effect of quote boundaries and punctuation on observed transitions.
- Compare counts and TF-IDF on a few short documents before interpreting a topic model.
- Practice explaining why a passing program can still produce incoherent language.

## Final summary

Week-8 connects text cleaning, dictionaries, probability, and testing through a simple text generator. Small hand-traced examples make the transition rules easier to understand, and retaining duplicates preserves observed word-pair frequencies. The completed class and seven passing tests provide evidence for the required behavior. My remaining work is to strengthen interpretation of preprocessing and topic-modeling choices, and to keep the distinction between successful sampling, readable language, and meaningful understanding clear.
