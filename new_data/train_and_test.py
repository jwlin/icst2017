"""
Controller to train and test data
"""

import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# local import
import util
import dirs
import core


if len(sys.argv) < 4:
    print('Missing arguments. e.g., python train_and_test.py 100 0.3 lsi-50-custom-stop-no-vote')

parsed_dir = dirs.parsed_dir
ratio_for_training = float(sys.argv[1])
repeated_times = int(sys.argv[2])
trial_name = sys.argv[3]
log_name = 't{:.1f}-r{:d}-{:s}'.format(ratio_for_training, repeated_times, trial_name)

util.setup_logger()
logger = util.get_logger()
util.add_file_logger(os.path.join('log', log_name))

if __name__ == '__main__':
    data = util.load_labeled_data(parsed_dir)
    data = [d for d in data if d['topic'] != 'unk']
    X = data
    y = [d['topic'] for d in data]

    res_pattern = []
    res_sim = []
    res_combine_n = []
    res_combine_m = []
    res_combine_b = []
    for i in range(repeated_times):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-ratio_for_training))

        # pattern-based method
        pattern_to_topic = core.train_by_pattern(X_train)
        y_pred_pattern = core.pred_by_pattern(X_test, pattern_to_topic)
        res_pattern.append(accuracy_score(y_test, y_pred_pattern))

        # similarity-based method
        dictionary, tfidf, lsi, index, idx_to_topic = core.train_by_sim(X_train)
        y_pred_sim = core.pred_by_sim(X_test, dictionary, tfidf, lsi, index, idx_to_topic)
        res_sim.append(accuracy_score(y_test, y_pred_sim))

        # combine
        y_pred_combine_n = core.pred_by_pattern_sim(X_test, pattern_to_topic, y_pred_sim, 'no-match')
        res_combine_n.append(accuracy_score(y_test, y_pred_combine_n))

        y_pred_combine_m = core.pred_by_pattern_sim(X_test, pattern_to_topic, y_pred_sim, 'multiple')
        res_combine_m.append(accuracy_score(y_test, y_pred_combine_m))

        y_pred_combine_b = core.pred_by_pattern_sim(X_test, pattern_to_topic, y_pred_sim, 'both')
        res_combine_b.append(accuracy_score(y_test, y_pred_combine_b))
        
        logger.info(
            'iteration {:d}: pattern: {:.4f}, sim: {:.4f}, combine_n: {:.4f}, combine_m: {:.4f}, combine_b: {:.4f}'.format(
                i, accuracy_score(y_test, y_pred_pattern), accuracy_score(y_test, y_pred_sim),
                accuracy_score(y_test, y_pred_combine_n), accuracy_score(y_test, y_pred_combine_m),
                accuracy_score(y_test, y_pred_combine_b)
        ))

        # ml-based method
        # train model on training data
        # test model on test data

    logger.info('accuracy: pattern: {:.4f}, sim: {:.4f}, combine_n: {:.4f}, combine_m: {:.4f}, combine_b: {:.4f}'.format(
        np.mean(res_pattern), np.mean(res_sim), np.mean(res_combine_n), np.mean(res_combine_m), np.mean(res_combine_b)
    ))
