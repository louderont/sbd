import re
from typing import List, Dict, Union


def word_tokenize(sentence:str) -> List[str]:
    """
    Split sentences into tokens separated by white space
    """
    return list(filter(lambda x: x not in [None, ' ', ''],
                       re.split(r' ', sentence)))


def contains_uppercase(token: Union[str, float]) -> bool:
    """
    Check whether the token contains an uppercase
    """
    if isinstance(token, str):
        return any(x.isupper() for x in token)
    else:
        return False


def contains_digit(token: Union[str, float]) -> bool:
    """
    Check whether the token contains an uppercase
    """
    if isinstance(token, str):
        return any(x.isdigit() for x in token)
    else:
        return False


def contains_period(token: Union[str, float]) -> bool:
    """
    Check whether the token contains a dot
    """
    if isinstance(token, str):
        return '.' in token
    else:
        return False


def contains_question_exclamation_mark(token: Union[str, float]) -> bool:
    """
    Check whether the token contains ? or !
    """
    if isinstance(token, str):
        return ('!' in token) | ('?' in token)
    else:
        return False


def contains_quote(token: Union[str, float]) -> bool:
    """
    Check whether the token contains quotes
    """
    if isinstance(token, str):
        return ("'" in token) | ('"' in token)
    else:
        return False
