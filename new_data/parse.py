"""
Manually label topics and write patterns for input fields of html pages,
then generate parsed input files from html pages
"""

import os
import json
import traceback
import sys
from bs4 import BeautifulSoup

# local import
import util

util.setup_logger()
logger = util.get_logger()
#util.add_file_logger('log.txt')

form_dir = 'forms'
parsed_dir = 'parsed'

topics = set()


if __name__ == '__main__':
    # https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input
    # added 'search', 'tel'
    input_types = ['text', 'email', 'password', 'search', 'tel']
    for html_file in os.listdir(form_dir):
        if os.path.isfile(os.path.join(form_dir, html_file)) and html_file.endswith('.html'):
            logger.info('Parsing %s', html_file)
            fname, ext = os.path.splitext(html_file)
            parsed_path = os.path.join(parsed_dir, fname + '.json')
            if os.path.exists(parsed_path):
                with open(parsed_path, 'r', encoding='utf-8') as pf:
                    data = json.load(pf)
                    topics = topics.union(set([d['topic'] for d in data]))
                continue
            with open(os.path.join(form_dir, html_file), 'rb') as f:
                dom = f.read().lower()
                soup = BeautifulSoup(dom, 'html5lib')
                tags_found = 0
                data = []
                for input_type in input_types:
                    order = 0
                    input_tags = soup.find_all('input', attrs={'type': input_type})
                    tags_found += len(input_tags)
                    for input_tag in input_tags:
                        try:
                            feature = util.extract_features(input_tag)
                            id_name = {k: v for k, v in input_tag.attrs.items() if k.lower() in ['id', 'name']}
                            topic = input('Label the input field:\nDOM (id and name): {:s}\nFeature: {:s}\n>> '.format(str(id_name), feature))
                            pattern = input('Topic "{:s}" is assigned. Write the identifying string or regex pattern '
                                            'for its id or name:\n>> '.format(topic))
                            assert util.is_validated(input_tag, pattern)
                            data.append({
                                'id': '{:s}-{:s}-{:d}'.format(fname, input_type, order),
                                'dom': str(input_tag),
                                'feature': feature,
                                'topic': topic,
                                'pattern': pattern
                            })
                            order += 1
                        except Exception as e:
                            logger.error('%s: Exception: %s', html_file, e)
                            traceback.print_exc(file=sys.stdout)
                            input('The pattern cannot match the id or name. Enter to continue')
                            continue
                assert tags_found > 2  # this should be first verified by generate_corpus.py
                new_topics = set([d['topic'] for d in data])
                logger.info('Newly added topics: %s', new_topics.difference(topics))
                topics = topics.union(new_topics)
                with open(parsed_path, 'w', encoding='utf-8') as pf:
                    json.dump(data, pf, ensure_ascii=False, sort_keys=True, indent=2)
                logger.info('Generate %s', parsed_path)

