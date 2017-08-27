import argparse
import json
import os

from utils.dataset import msg2txt, random_batch_interleaving, split
from utils.encoder import Encoder
from utils.preprocessor import Preprocessor


def download_messages(token, chat_room, msg_path):
    """
    Batch downloads all history of a given gitter room
    Args:
        token: the gitter auth token
        chat_room: the target chat room
        msg_path: download older than this message
    """
    if token is None:
        raise ValueError("Give a valid gitter API token.")
    else:
        from utils.gitter import get_all_messages
        get_all_messages(token, chat_room, msg_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('seq_len', type=int)
    parser.add_argument('batch_size', type=int)
    parser.add_argument('--gitter_token', type=str, default=None,
                        help="The gitter token")
    parser.add_argument('--chat_room', type=str, default="tiny-dnn/developers",
                        help="If not set to json file, a gitter token must be given to download them")
    parser.add_argument('--msg_path', type=str, default="../data/messages.json",
                        help="If not set to json file, a gitter token must be given to download them")
    parser.add_argument('--encoding_file', type=str, default=None,
                        help="If not set to json file, it will be computed")
    parser.add_argument('--train_split', type=float, default=0.9)
    parser.add_argument('--train_output', type=str, default="../data/train.raw",
                        help="The path to output the train data")
    parser.add_argument('--val_output', type=str, default="../data/val.raw",
                        help="The path to output the validation data")
    parser.add_argument('--max_train_size', type=int, default=1e6)
    parser.add_argument('--max_val_size', type=int, default=0)
    args = parser.parse_args()

    if not (os.path.isfile(args.msg_path)):
        print("Downloading from gitter...")
        download_messages(args.gitter_token, args.chat_room, args.msg_path)

    with open(args.msg_path, 'r') as input:
        print("Loading messages form disk...")
        messages = json.load(input)

    preprocessor = Preprocessor()
    print("Preprocessing...")
    for idx, message in enumerate(messages):
        messages[idx]['text'] = preprocessor.process_text(message['text'])
        messages[idx]['fromUser']['username'] = preprocessor.process_text(message['fromUser']['username'],
                                                                          newline=False)

    encoder = Encoder()
    if args.encoding_file is None:
        print("Generating encoding dictionary...")
        encoder.gen_dict(msg2txt(messages))
        encoder.save_enc_dict_json(path='../data/encoding.json')
        encoder.save_dec_dict_binary(path='../data/encoding.raw')
    else:
        print("Loading encoding dictionary from disk...")
        encoder.load_dict(args.encoding_file)

    print("Creating random train/val splits...")
    train, val = split(messages, chunks=5, train_ratio=args.train_split)
    print("Batch interleaving...")
    if args.max_train_size == 0:
        args.max_train_size = len(train) ** 2
    if args.max_val_size == 0:
        args.max_val_size = len(val) ** 2

    x_t, y_t = random_batch_interleaving(train, args.batch_size, args.seq_len, args.max_train_size)
    x_v, y_v = random_batch_interleaving(val, args.batch_size, args.seq_len, args.max_val_size)

    print("Encoding and saving to disk...")
    encoder.encode(x_t, args.train_output)
    encoder.encode(y_t, args.train_output.split('.')[0] + '_labels.raw')
    encoder.encode(x_v, args.val_output)
    encoder.encode(y_v, args.val_output.split('.')[0] + '_labels.raw')
