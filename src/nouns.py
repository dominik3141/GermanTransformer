"""
Defines the Noun class, which represents a noun together with its article.
"""


class Noun:
    """
    A noun is a string together with an article
    """

    def __init__(self, word: str, article: str):
        self.word = word
        self.article = article  # "m", "f", "n"

    def __str__(self):
        return f"{self.word}, {self.article}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.word == other.word and self.article == other.article
