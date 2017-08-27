import json

from gitterpy.client import GitterClient, BaseApi


def get_all_messages(gitter_token="",
                     chat="tiny-dnn/developers", out_path='messages.json'):
    gitter = GitterClient(gitter_token)

    messages = []
    ret = gitter.messages.list(chat)
    messages += ret
    while len(ret) != 0:
        before_id = messages[0]['id']
        BaseApi.footers = "?limit=100&beforeId=%s" % str(before_id)
        try:
            ret = gitter.messages.list(chat)
        except:
            break
        messages = ret + messages
        if len(ret) > 0:
            print(ret[0])

    with open(out_path, 'w') as output:
        json.dump(messages, output)
