import pytest
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sbdetection.preprocessing.utils import word_tokenize, contains_uppercase, contains_period, \
    contains_question_exclamation_mark, contains_quote
from sbdetection.preprocessing.preprocessor import sentences_to_binary_target, binary_target_to_sentences

TEST_TOKENIZER = [
    ('Hello world. My name is X.', [
     'Hello', 'world.', 'My', 'name', 'is', 'X.']),
    ('This is dr$ X speaking!', [
     'This', 'is', 'dr$', 'X', 'speaking!'])
]

TEST_CONTAINS_UPPERCASE = [
    ('Hello', True), ('HellO.', True),
    ('test', False), (np.nan, False),
    ('.', False)
]

TEST_CONTAINS_PERIOD = [
    ('Hello', False), ('.', True),
    ('test.', True), (np.nan, False)
]

TEST_CONTAINS_MARKS = [
    ('Hello', False), ('?!', True),
    ('test!', True), (np.nan, False),
    ('?', True)
]

TEST_CONTAINS_QUOTE = [
    ('Hello', False), ('"', True),
    ('"test"', True), (np.nan, False),
    ("'", True), ("''", True),
    ('""', True)
]

TEST_STOPWORDS = [
    (('The'), True),
    (('a'), True),
    (('sometimes'), False),
    (('code'), False)
]

TEST_SENTENCES_TO_TARGET = [
    (['This is. the. test.'], np.array([0, 0, 0, 1])),
    (['.'], np.array([1]))
]

TEST_REMOVE_SPACE = [
    ("Hello !", "Hello !"),
    ("How are you ?!", "How are you ?!")
]

TEST_TARGET_TO_SENTENCES = [
    (pd.DataFrame({'token': ['Hello', '!', 'Are', 'you', 'ok?'],
                  'target':[0, 1, 0, 0, 1]}),
     ['Hello !', 'Are you ok?']),
    (pd.DataFrame({'token': ['Are', 'you', 'ok', '!?'],
                   'target': [0, 0, 0, 1]}),
     ['Are you ok !?']),
]


@ pytest.mark.parametrize(('sentence, expected_tokens'), TEST_TOKENIZER)
def test_word_tokenizer(sentence, expected_tokens):
    tokens = word_tokenize(sentence)
    assert tokens == expected_tokens


@ pytest.mark.parametrize(('token, expected_bool'), TEST_CONTAINS_UPPERCASE)
def test_contains_uppercase(token, expected_bool):
    contains_upper = contains_uppercase(token)
    assert contains_upper == expected_bool


@ pytest.mark.parametrize(('token, expected_bool'), TEST_CONTAINS_PERIOD)
def test_contains_period(token, expected_bool):
    res_contains_upper = contains_period(token)
    assert res_contains_upper == expected_bool


@ pytest.mark.parametrize(('token, expected_bool'), TEST_CONTAINS_MARKS)
def test_contains_mark(token, expected_bool):
    res_contains_mark = contains_question_exclamation_mark(token)
    assert res_contains_mark == expected_bool


@ pytest.mark.parametrize(('token, expected_bool'), TEST_CONTAINS_QUOTE)
def test_contains_quote(token, expected_bool):
    res_contains_quote = contains_quote(token)
    assert res_contains_quote == expected_bool


@ pytest.mark.parametrize(('sentence, expected_array'), TEST_SENTENCES_TO_TARGET)
def test_sentences_to_binary_target(sentence, expected_array):
    res_target = sentences_to_binary_target(sentence)
    assert np.array_equal(res_target['target'].values, expected_array)


@ pytest.mark.parametrize(('sentence, expected_sentence'), TEST_TARGET_TO_SENTENCES)
def test_binary_target_to_sentences(sentence, expected_sentence):
    res_sentences = binary_target_to_sentences(sentence)
    assert all([a == b for a, b in zip(res_sentences, expected_sentence)])
