import os
import json

parsed_dir = 'parsed'

if __name__ == '__main__':
    topics = set()
    for json_file in os.listdir(parsed_dir):
        if os.path.isfile(os.path.join(parsed_dir, json_file)) and json_file.endswith('.json'):
            with open(os.path.join(parsed_dir, json_file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                topics = topics.union(set([d['topic'] for d in data]))
    print('All {:d} topics:'.format(len(topics)))
    for t in sorted(list(topics)):
        print(t)
