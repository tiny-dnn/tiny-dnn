# -*- coding: utf-8 -*-

class Preprocessor(object):
    """
    Reduces the amount of chars to a small subset
    """
    def __init__(self, replacements=None):
        """
        Constructor
        Args:
            replacements: dictionary of chars to replace
        """
        if replacements is None:
            self.replacements = {
                'à': 'a',
                'á': 'a',
                'è': 'e',
                'é': 'e',
                'ì': 'i',
                'í': 'i',
                'ò': 'o',
                'ó': 'o',
                'ù': 'u',
                'ú': 'u',
                'ç': 'c',
                'ñ': 'n',
                '\n': ''
            }
        else:
            self.replacements = replacements

    def __collapse_spaces(self, text):
        """
        Collapses multiple spces into a single one
        Args:
            text: string to process

        Returns: collapsed text.

        """
        while "  " in text:
            text = text.replace("  ", " ")
        return text

    def __replace_special(self, text):
        """
        Replaces special chars
        Args:
            text: input string

        Returns: processed string

        """
        for key in self.replacements:
            text = text.replace(key, self.replacements[key])
        return text

    def process_text(self, text, newline=True):
        """
        Removes special chars, collapses, etc.
        Args:
            text: input string
            newline: whether to finish strings with newline char

        Returns:

        """
        buffer = ""
        text = self.__collapse_spaces(self.__replace_special(text).lower())
        for t in text:
            ch = ord(t)
            if ch < 32 or ch > 126:
                continue
            else:
                buffer += t
        if newline:
            buffer += '\n'
        return buffer
