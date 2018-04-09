"""
Generate corpus files from html pages
"""

import os
from bs4 import BeautifulSoup

# local import
import util

util.setup_logger()
logger = util.get_logger()

form_dir = 'forms'
corpus_dir = 'corpus'


if __name__ == '__main__':
    input_types = ['text', 'email', 'password']
    for html_file in os.listdir(form_dir):
        if os.path.isfile(os.path.join(form_dir, html_file)) and html_file.endswith('.html'):
            fname, ext = os.path.splitext(html_file)
            corpus_path = os.path.join(corpus_dir, fname + '.corpus')
            try:
                os.remove(corpus_path)
            except OSError:
                pass
            with open(os.path.join(form_dir, html_file), 'r', encoding='utf-8') as f:
                dom = f.read().lower()
                soup = BeautifulSoup(dom, 'html5lib')
                for input_type in input_types:
                    for input_tag in soup.find_all('input', attrs={'type': input_type}):
                        with open(corpus_path, 'a') as cf:
                            cf.write(util.extract_features(input_tag) + '\n')
            logger.info('Generate %s', corpus_path)