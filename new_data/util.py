import re
import sys
import logging


# Usage of logger:
# 1. util.setup_logger() at first
# 2. For setting file logger, call util.add_file_logger(fname)
# 3. In other codes, use logger by logging.getLogger(util.logger_name)
#                                  logger.setLevel(logging.DEBUG)
#                                  logger.debug('Bug'), etc.
logger_name = 'ityc'
formatter = logging.Formatter("[%(asctime)s][%(filename)s:%(lineno)s][%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")


def setup_logger():
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)


def get_logger():
    return logging.getLogger(logger_name)


def add_file_logger(fname):
    logger = logging.getLogger(logger_name)
    file_handler = logging.FileHandler(fname)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)


# generate feature vector for each input field from html doc
def extract_features(soup_element):
    descriptive_attrs = ['id', 'name', 'value', 'type', 'placeholder']
    constraint_attrs = ['maxlength']
    feature_d = []  # descriptive features of the input element
    feature_c = []  # constraint features of the input element
    label_features = find_closest_labels(soup_element, iteration=5)
    if label_features:
        feature_d += label_features
    for key, value in soup_element.attrs.items():
        if value and key in descriptive_attrs:
            value = re.sub('[^a-zA-Z0-9]', ' ', value.lower())
            feature_d += [value]
        if value and key in constraint_attrs:
            value = re.sub('[^a-zA-Z0-9]', ' ', value.lower())
            feature_c += [value]
    return re.sub(' +', ' ', ' '.join(feature_d + feature_c)).strip()


# find the  closest spans/labels of a DOM and return the text
def find_closest_labels(soup_element, iteration):
    if iteration == 0:  # No label found after multiple iterations
        return None
    siblings = []
    siblings += soup_element.find_previous_siblings()
    siblings += soup_element.find_next_siblings()
    labels = []
    candidate_tags = ['span', 'label']
    for tag in candidate_tags:
        labels += soup_element.find_previous_siblings(tag)
        labels += soup_element.find_next_siblings(tag)
        for sib in siblings:
            labels += sib.find_all(tag)
    if not labels:
        if soup_element.name.lower() == 'form':  # stop. We don't want to step out the form
            return None
        return find_closest_labels(soup_element.parent, iteration-1) if soup_element.parent else None
    else:
        content = []
        for l in labels:
            for s in l.stripped_strings:
                content.append(re.sub('[^a-zA-Z0-9]', ' ', s.lower()))
        if content:
            return content
        if soup_element.name.lower() == 'form':  # stop. We don't want to step out the form
            return None
        return find_closest_labels(soup_element.parent, iteration - 1) if soup_element.parent else None


# validate the regex can be used to identify the dom by its id or name
def is_validated(soup_element, pattern):
    is_matched = False
    for key, value in soup_element.attrs.items():
        if key.lower() in ['id', 'name']:
            if re.search(pattern, value):
                is_matched = True
                break
    return is_matched
