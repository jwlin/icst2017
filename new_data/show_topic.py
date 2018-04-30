import os
import json
from bs4 import BeautifulSoup
from collections import defaultdict


parsed_dir = 'parsed'

if __name__ == '__main__':
    topic = defaultdict(int)
    for json_file in os.listdir(parsed_dir):
        if os.path.isfile(os.path.join(parsed_dir, json_file)) and json_file.endswith('.json'):
            with open(os.path.join(parsed_dir, json_file), 'r', encoding='utf-8') as f:
                data = json.load(f)                
                for d in data:
                    if d['topic'] == '':
                        input_tag = BeautifulSoup(d['dom'], 'html5lib').find('input')
                        tag_id = input_tag['id'] if 'id' in input_tag.attrs else None
                        tag_name = input_tag['name'] if 'name' in input_tag.attrs else None
                        print('Tag uid: {}, topic: {}, id: {}, name: {}'.format(d['id'], d['topic'], tag_id, tag_name))
                    topic[d['topic']] += 1
    print('All {:d} topics:'.format(len(topic)))
    print('All {:d} input fields:'.format(sum(topic.values())))
    t = [(k, v) for k, v in topic.items()]
    #t = sorted(t, key=lambda x: x[1], reverse=True)
    t = sorted(t, key=lambda x: x[0])
    for k, v in t:
        print(k, v)
