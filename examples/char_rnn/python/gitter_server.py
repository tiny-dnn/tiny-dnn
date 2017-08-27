from __future__ import print_function, absolute_import

import argparse
import json
import pycurl
from io import BytesIO

import tiny_dnn.tiny_char_rnn as char_rnn
from utils.preprocessor import Preprocessor

""" Gitter Interface
Helper functions to read/write from a gitter room.
"""
class GitterInterface(object):
    def __init__(self, token, room, char_rnn):
        self.char_rnn = char_rnn
        self.resturl = "https://api.gitter.im/v1/rooms/"
        self.restheader = ["Content-Type: application/json", "Accept: application/json",
                           "Authorization: Bearer %s" % token]
        self.buffer = []
        # Get room id
        buffer = BytesIO()
        data = json.dumps({'uri': room.lower()})
        conn = pycurl.Curl()
        conn.setopt(pycurl.URL, self.resturl)
        conn.setopt(pycurl.HTTPHEADER, self.restheader)
        conn.setopt(pycurl.POST, 1)
        conn.setopt(pycurl.POSTFIELDS, data)
        conn.setopt(pycurl.WRITEDATA, buffer)
        conn.perform()
        conn.close()
        self.room_id = json.loads(buffer.getvalue().decode('utf-8'))['id']
        self.streamingurl = "https://stream.gitter.im/v1/rooms/%s/chatMessages" % self.room_id
        self.streamheader = ["Accept: application/json", "Authorization: Bearer %s" % token]
        # Send welcome message
        self.send("**tiny_char_rnn bot:** say something and I'll try to answer :D\n Usage: @tiny_char_rnn query")

    """
    Main method
    """
    def run(self):
        # Start listening to the gitter streaming API
        conn = pycurl.Curl()
        conn.setopt(pycurl.URL, self.resturl)
        conn.setopt(pycurl.HTTPHEADER, self.streamheader)
        conn.setopt(pycurl.URL, self.streamingurl)
        conn.setopt(pycurl.WRITEFUNCTION, self._callback)
        conn.perform()

    """
    Send string message to room. 
    """
    def send(self, text):
        data = json.dumps({'text': text})
        conn = pycurl.Curl()
        conn.setopt(pycurl.URL, "%s/%s/chatMessages" % (self.resturl, self.room_id))
        conn.setopt(pycurl.HTTPHEADER, self.restheader)
        conn.setopt(pycurl.POST, 1)
        conn.setopt(pycurl.POSTFIELDS, data)
        conn.perform()
        conn.close()

    """
    Streaming API callback. 
    """
    def _callback(self, data):
        if data != b' \n':
            d = json.loads(data.decode('utf-8'))
            if "@tiny_char_rnn" in d["text"]:
                text = "%s\t%s\n" % (d["fromUser"]["username"], d["text"])
                self.char_rnn.set_input(text)
                self.send("**tiny_char_rnn**: %s" % self.char_rnn.get_output())


""" TinyCharRNN
Handles tiny_dnn wrapper I/O
"""
class CharRNN(object):
    def __init__(self, weights, encoding, rnn_type, depth, hidden_size, softmax_temp=0.9, output_lim=144):
        self.preprocessor = Preprocessor()
        self.model = char_rnn.Model(weights, encoding, rnn_type, depth, hidden_size)
        self.output_lim = output_lim
        self.temperature = softmax_temp
        self.buffer = '\n'

    def set_input(self, text):
        if len(text) > 0:
            data = self.preprocessor.process_text(text, newline=False)
            if len(data) == 0 or data[-1] == '\n':
                data = '\n'
            for c in data:
                self.buffer = str(self.model.forward(c, self.temperature))
        else:
            self.buffer = self.buffer[-1]

    def get_output(self):
        for i in range(self.output_lim - 1):
            c = self.buffer[-1]
            self.buffer += str(self.model.forward(c, self.temperature))
            if self.buffer[-1] == '\n':
                break
        return self.buffer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("gitter_room", type=str,
                        help="Room to interact with. If local, stdin/stdout will be used")
    parser.add_argument("gitter_api_token", type=str,
                        help="A gitter API token")
    parser.add_argument("--weights_path", "-w", type=str, default="../data/char_rnn_weights",
                        help="Weights to load the rnn.")
    parser.add_argument("--encoding_path", "-e", type=str, default="../data/encoding.raw",
                        help="Path for the text encoding dictionary.")
    parser.add_argument("--depth", "-d", type=int, default=3,
                        help="Depth of the pre-trained model")
    parser.add_argument("--hidden_size", "-s", type=int, default=256,
                        help="Recurrent state size.")
    parser.add_argument("--rnn_type", "-r", type=str, default="gru", choices=["gru", "lstm", "rnn"],
                        help="Pre-trained RNN cell type.")
    parser.add_argument("--max_output_size", type=int, default=200,
                        help="Max number of chars to predict.")
    parser.add_argument("--softmax_temp", type=float, default=0.5,
                        help="Softmax temperature, the higher, the most unexpected will be the output.")
    args = parser.parse_args()

    tiny_char_rnn = CharRNN(args.weights_path, args.encoding_path, args.rnn_type, args.depth, args.hidden_size,
                            args.softmax_temp, args.max_output_size)
    gitter = GitterInterface(args.gitter_api_token, args.gitter_room, tiny_char_rnn)
    gitter.run()
