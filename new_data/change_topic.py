import os
import json

parsed_dir = 'parsed'

if __name__ == '__main__':
    old_topic = input('The topic name you want to change: ')
    new_topic = input('Change {:s} to: '.format(old_topic))
    for json_file in os.listdir(parsed_dir):
        is_changed = False
        if os.path.isfile(os.path.join(parsed_dir, json_file)) and json_file.endswith('.json'):
            parsed_path = os.path.join(parsed_dir, json_file)
            with open(parsed_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for d in data:
                    if d['topic'] == old_topic:
                        d['topic'] = new_topic
                        is_changed = True
            if is_changed:
                with open(parsed_path, 'w', encoding='utf-8') as pf:
                    json.dump(data, pf, ensure_ascii=False, sort_keys=True, indent=2)
                is_changed = False
                print('Changed {}'.format(json_file))