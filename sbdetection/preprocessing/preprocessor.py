from dataclasses import dataclass
from typing import Dict, List
import pandas as pd

from nltk.corpus import stopwords
from .utils import (
    contains_digit, contains_period, 
    contains_question_exclamation_mark,
    contains_quote, contains_uppercase,
    word_tokenize)
    
def feature_extraction(df: pd.DataFrame, shifts: Dict[str, int]) -> pd.DataFrame:
    """
    Feature extraction adding shifted columns to df

    Args:
        df: dataframe with a "token" column containing tokens parsed using sbdetection.utils.word_tokenize
        shifts: shift's name (str): number of shift to apply to df['token']
    Returns:
        df_feature:
    """
    df_feature = df.copy()
    stop_words_nltk = [word for word in stopwords.words('english')]

    # feature's name (str): function to apply to df['token'] to obtain the feature (callable)
    function_features = {"contains_uppercase": contains_uppercase,
                         "contains_?_!": contains_question_exclamation_mark,
                         "contains_period": contains_period,
                         "contains_quote": contains_quote,
                         "contains_digit": contains_digit,
                         "token_length": lambda x: len(x) if isinstance(x, str) else False
                         }

    for shift_name, shift in shifts.items():
        for func_name, func in function_features.items():
            df_feature.loc[:, f'{func_name}_{shift_name}'] = df['token'].shift(
                shift).apply(func).astype('int')
        # add an - is stopword - feature
        df_feature.loc[:, f'is_stopword_{shift_name}'] = df['token'].shift(
            shift).isin(stop_words_nltk).astype('int')
    return df_feature

def sentences_to_binary_target(sentences: List[str]) -> pd.DataFrame:
    """
    Generate a binary target describing tokens from sentences

    Args:
        sentences: each element is a sentence
    Returns:
        dataframe with one token per row, 2 columns:
            - token: string value
            - target: whether the token is sentence ending token
    """
    sublist_token = [word_tokenize(sentence) for sentence in sentences]
    is_the_end = sum([[0]*(len(tokens)-1)+[1] for tokens in sublist_token], [])
    return pd.DataFrame(
        {'target': is_the_end, 'token': sum(sublist_token, [])})


def binary_target_to_sentences(df: pd.DataFrame) -> List[str]:
    """
    Generate sentences from a binary target describing tokens

    Args:
        dataframe with one token per row, 2 columns:
            - token: string value
            - target: whether the token is sentence ending token
    Returns:
        sentences: each element is a sentence
    """
    # compute a cumsum on the binary target serie in reversed order to get a sentence id
    sentences = (
        df.assign(id=df.loc[::-1, 'target'].cumsum()
                  ).groupby('id')['token'].apply(' '.join)[::-1]
    ).values
    return sentences
