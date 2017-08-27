import numpy as np


def split(messages, chunks=5, train_ratio=0.9):
    length = chunks * (len(messages) // chunks)
    indices = np.arange(length).reshape(-1, chunks)
    chunk_indices = np.arange(indices.shape[0])
    np.random.shuffle(chunk_indices)
    n_train = int(np.ceil(chunk_indices.shape[0] * train_ratio))
    train_indices = np.sort(indices[chunk_indices[:n_train], :].ravel())
    val_indices = np.sort(indices[chunk_indices[n_train:], :].ravel())

    train = []
    val = []

    for idx in train_indices:
        train.append(messages[idx])
    for idx in val_indices:
        val.append(messages[idx])

    return train, val


def msg2txt(messages):
    ret = ""
    for m in messages:
        user = m['fromUser']['username']
        text = m['text']
        ret += user + '\t' + text
    return ret


def random_batch_interleaving(messages, batch_size, seq_len, max_size=0):
    text = msg2txt(messages)
    if max_size == 0:
        max_size = len(text)**2
    input = ""
    output = ""
    while len(input) < max_size:
        print('%.02f' %(100 * float(len(input)) / max_size), '%')
        batches = np.random.randint(0, len(text) - seq_len - 1, batch_size)
        for s in range(seq_len):
            for i in batches:
                input += text[i]
                output += text[i + 1]
            batches += 1

    return input, output

