import io
import json


class Encoder(object):
    """
    Encodes strings to an array of contiguous numbers
    """
    def __init__(self):
        """
        Constructor
        """
        self.enc_dict = None
        self.dec_dict = None

    def load_dict(self, path):
        """
        Loads json dictionary to encode strings.
        Args:
            path: string path of the dictionary.
        """
        with open(path, 'r') as infile:
            self.enc_dict = json.load(infile)
        self.dec_dict = [0] * len(self.enc_dict.keys())
        for key in self.enc_dict.keys():
            self.dec_dict[self.enc_dict[key]] = key

    def gen_dict(self, text):
        """
        Generates a dictionary of contiguous numbers to encode the text.
        Args:
            text: input string text
        """
        self.enc_dict = {}
        self.dec_dict = []
        for t in sorted(text):
            if t not in self.enc_dict:
                self.enc_dict[t] = len(self.dec_dict)
                self.dec_dict.append(t)

    def save_enc_dict_json(self, path='encoding.json'):
        """
        Saves the encoding dict in json format
        Args:
            path: string output path
        """
        with open(path, 'w') as output:
            json.dump(self.enc_dict, output)

    def save_dec_dict_binary(self, path='encoding.raw'):
        """
        Saves the positional decoding dict in raw format.
        Args:
            path: string output path
        """
        with io.FileIO(path, 'w') as out:
            for c in self.dec_dict:
                out.write(bytes([ord(c)]))

    def encode(self, input, output):
        """
        Encodes input with the dictionary
        Args:
            input: input string to encode
            output: string output path
        """
        assert (len(self.dec_dict) > 0 and len(self.enc_dict.keys()) > 0)
        with io.FileIO(output.split('.')[0] + '.raw', 'w') as out:
            for c in input:
                out.write(bytes([self.enc_dict[c]]))

    def decode(self, input, output=None):
        """
        Decodes input with the dictionary
        Args:
            input: input file path to decode
            output: string output file path.

        Returns: If output is None returns the decoded text

        """
        with io.FileIO(input, 'r') as infile:
            stream = infile.read()
        text = ""

        for s in stream:
            text += self.dec_dict[s]

        if output is not None:
            with open(output, 'w') as out:
                out.write(text)

        return text
