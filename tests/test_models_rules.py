import pytest

from sbdetection.models import baseline

TEST_RULE_BASED_DOCUMENTS = [
    ('Hello world. My name is C.', ['Hello world.', 'My name is C.']),
    ('This is C speaking.', ['This is C speaking.']),
    ('This is dr. John speaking.', ['This is dr$ John speaking.']),
    ('He is a DR. in physics in some time.', [
     'He is a DR$ in physics in some time.']),
    ('I have a PhD. in math.', ['I have a PhD$ in math.']),
    ('The website: https://www.example.com.html. The content is there.',
     ['The website: https://www.example.com.html.', 'The content is there.'])
]


@pytest.mark.parametrize('document, expected_sentences', TEST_RULE_BASED_DOCUMENTS)
def test_baseline(document, expected_sentences):
    sentences = baseline.predict_sentences(document)
    assert sentences == expected_sentences
