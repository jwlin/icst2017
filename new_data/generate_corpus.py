"""
Generate corpus files from html pages
"""

import os
from bs4 import BeautifulSoup

# local import
import util

util.setup_logger()
logger = util.get_logger()
util.add_file_logger('log.txt')

form_dir = 'forms'
corpus_dir = 'corpus'


if __name__ == '__main__':
    # https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input
    # added 'search', 'tel'
    input_types = ['text', 'email', 'password', 'search', 'tel']
    for html_file in os.listdir(form_dir):
        if os.path.isfile(os.path.join(form_dir, html_file)) and html_file.endswith('.html'):
            fname, ext = os.path.splitext(html_file)
            corpus_path = os.path.join(corpus_dir, fname + '.corpus')
            try:
                os.remove(corpus_path)
            except OSError:
                pass
            with open(os.path.join(form_dir, html_file), 'rb') as f:
                try:
                    dom = f.read().lower()
                    soup = BeautifulSoup(dom, 'html5lib')
                    tags_found = 0
                    for input_type in input_types:
                        input_tags = soup.find_all('input', attrs={'type': input_type})
                        tags_found += len(input_tags)
                        for input_tag in input_tags:
                            with open(corpus_path, 'a', encoding='utf-8') as cf:
                                cf.write(util.extract_features(input_tag) + '\n')
                    if tags_found > 2:
                        logger.info('Generate %s', corpus_path)
                    else:
                        logger.warning('%s: %d tags found', html_file, tags_found)
                except Exception as e:
                    logger.error('%s: Exception: %s', html_file, e)
