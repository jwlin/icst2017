from bs4 import BeautifulSoup
from collections import defaultdict

import dirs
from util import load_labeled_data

parsed_dir = dirs.parsed_dir

if __name__ == '__main__':
    topic = defaultdict(int)
    data = load_labeled_data(parsed_dir)
    for d in data:
        if d['topic'] == '':
        #if d['topic'] in ['unk']:
            input_tag = BeautifulSoup(d['dom'], 'html5lib').find('input')
            tag_id = input_tag['id'] if 'id' in input_tag.attrs else None
            tag_name = input_tag['name'] if 'name' in input_tag.attrs else None
            print('Tag uid: {}, topic: {}, id: {}, name: {}'.format(d['id'], d['topic'], tag_id, tag_name))
            print('Feature: {}'.format(d['feature']))
        topic[d['topic']] += 1
    print('All {:d} topics:'.format(len(topic)))
    print('All {:d} input fields:'.format(sum(topic.values())))
    t = [(k, v) for k, v in topic.items()]
    t = sorted(t, key=lambda x: x[1], reverse=True)
    #t = sorted(t, key=lambda x: x[0])
    for k, v in t:
        print(k, v)
