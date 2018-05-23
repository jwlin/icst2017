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
import rnn


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
    res_nb = []
    res_svm = []
    res_logit = []
    res_rf = []
    res_rnn = []
    for i in range(repeated_times):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-ratio_for_training))

        '''
        # pattern-based method
        pattern_to_topic = core.train_by_pattern(X_train)
        y_pred_pattern = core.pred_by_pattern(X_test, pattern_to_topic)
        res_pattern.append(accuracy_score(y_test, y_pred_pattern))

        # similarity-based method
        dictionary, tfidf, lsi, index, idx_to_topic = core.train_by_sim(X_train)
        y_pred_sim = core.pred_by_sim(X_test, dictionary, tfidf, lsi, index, idx_to_topic)
        res_sim.append(accuracy_score(y_test, y_pred_sim))
        '''

        '''
        # combine
        y_pred_combine_n = core.pred_by_pattern_sim(X_test, pattern_to_topic, y_pred_sim, 'no-match')
        res_combine_n.append(accuracy_score(y_test, y_pred_combine_n))

        y_pred_combine_m = core.pred_by_pattern_sim(X_test, pattern_to_topic, y_pred_sim, 'multiple')
        res_combine_m.append(accuracy_score(y_test, y_pred_combine_m))

        y_pred_combine_b = core.pred_by_pattern_sim(X_test, pattern_to_topic, y_pred_sim, 'both')
        res_combine_b.append(accuracy_score(y_test, y_pred_combine_b))
        '''

        '''
        # ml-based
        classifier = core.train_by_ml(X_train, model='nb')
        y_pred_nb = core.pred_by_ml(classifier, X_test)
        res_nb.append(accuracy_score(y_test, y_pred_nb))

        classifier = core.train_by_ml(X_train, model='svm')
        y_pred_svm = core.pred_by_ml(classifier, X_test)
        res_svm.append(accuracy_score(y_test, y_pred_svm))

        classifier = core.train_by_ml(X_train, model='logit')
        y_pred_logit = core.pred_by_ml(classifier, X_test)
        res_logit.append(accuracy_score(y_test, y_pred_logit))

        classifier = core.train_by_ml(X_train, model='rf')
        y_pred_rf = core.pred_by_ml(classifier, X_test)
        res_rf.append(accuracy_score(y_test, y_pred_rf))
        '''

        model, token_vocab, le = rnn.train(X_train, y_train)
        y_pred_rnn = rnn.pred(model, token_vocab, le, X_test)
        res_rnn.append(accuracy_score(y_test, y_pred_rnn))

        logger.info(
            'iteration {:d}: rnn: {:.4f}'.format(
                i, accuracy_score(y_test, y_pred_rnn))
        )

        '''
        logger.info(
            'iteration {:d}: pattern: {:.4f}, sim: {:.4f}, nb: {:.4f}, svm: {:.4f}, logit: {:.4f}, rf: {:.4f}'.format(
                i, accuracy_score(y_test, y_pred_pattern), accuracy_score(y_test, y_pred_sim),
                accuracy_score(y_test, y_pred_nb), accuracy_score(y_test, y_pred_svm),
                accuracy_score(y_test, y_pred_logit), accuracy_score(y_test, y_pred_rf))
        )
        '''

    logger.info(
        'accuracy: rnn: {:.4f}'.format(
            np.mean(res_rnn))
    )
    #logger.info('accuracy: pattern: {:.4f}, sim: {:.4f}, nb: {:.4f}, svm: {:.4f}, logit: {:.4f}, rf: {:.4f}'.format(
    #    np.mean(res_pattern), np.mean(res_sim), np.mean(res_nb), np.mean(res_svm), np.mean(res_logit), np.mean(res_rf)
    #))
