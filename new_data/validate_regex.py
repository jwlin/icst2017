import os
import json
from bs4 import BeautifulSoup
from util import is_validated


parsed_dir = 'parsed'

if __name__ == '__main__':
    count = 0
    for json_file in os.listdir(parsed_dir):
        if os.path.isfile(os.path.join(parsed_dir, json_file)) and json_file.endswith('.json'):
            with open(os.path.join(parsed_dir, json_file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                for d in data:
                    input_tag = BeautifulSoup(d['dom'], 'html5lib').find('input')
                    if not is_validated(input_tag, d['pattern']) and d['topic'] != 'unk':
                        tag_id = input_tag['id'] if 'id' in input_tag.attrs else None
                        tag_name = input_tag['name'] if 'name' in input_tag.attrs else None
                        print('Tag uid: {}, id: {}, name: {}'.format(d['id'], tag_id, tag_name))
                        print('pattern:', d['pattern'])
                        count += 1
    print('# of mismatched tags:', count)
