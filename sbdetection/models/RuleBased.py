from typing import List
import re


class Baseline:
    """Baseline implements basic rule based SBD.

    It is based on the golden rules: https://s3.amazonaws.com/tm-town-nlp-resources/golden_rules.txt
    and implements the first 9 rules to detect sentence boundary.

    """

    def predict_sentences(self, document: str) -> List[str]:
        """apply the golden rules thanks to regex and return the sentences

        Args:
            document str: a group of sentences that we want to split in several sentences

        Returns:
            sentences List[str]: the sentences extracted from the document    
        """

        # we replace abbreviation
        document = re.sub(r'(\b)([A-Za-z]{1,3})(\.)( )', r'\1\2$\4', document)
        document = re.sub(r'(\b)([0-9]{1,3})(\.)( )', r'\1\2$\4', document)

        # split sentences with . or ? or ! for ending words.
        sentences = re.split(r'\. |! |\? ', document)
        # # add . at the end of sentences
        sentences = [
            f'{sentence.strip()}.' for sentence in sentences[:-1]] + [sentences[-1]]
        return sentences

    def predict(self, data: List[str]) -> List[str]:
        """generate sentence boundary from a list a documents

        Args:
            data List[str]: a list of document on which to run the SBD 

        Returns:
            parsed_data List[str]: for each document, the sentences are computed
        """
        parsed_data = [self.predict_sentences(document) for document in data]
        output = sum(parsed_data, [])
        return output


baseline = Baseline()
