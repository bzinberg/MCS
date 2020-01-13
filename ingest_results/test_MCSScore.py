import sys
import traceback
from unittest import TestCase
import numpy as np

from sklearn.metrics import roc_auc_score

from mcs_score import MCSScore


class TestMCSScore(TestCase):

    def test_loss_scene_relative(self):
        scorer = MCSScore()

        ground_truth = {'1': 0, '2': 1, '3': 1, '4': 0}
        good = {'1': 0.2, '2': 0.6, '3': 0.7, '4': 0.3}

        loss = scorer.loss_scene_relative(good, ground_truth)
        print("Loss of good is : {}".format(loss))

        gt_by_scene = {"1": ground_truth}
        good_by_scene = {"1": good}
        loss2 = scorer.loss_relative(good_by_scene, gt_by_scene)
        print("Loss2 of good is : {}".format(loss2))

        bad = {'1': 0.9, '2': 0.1, '3': 0.3, '4': 0.8}
        loss = scorer.loss_scene_relative(bad, ground_truth)
        print("Loss of bad is : {}".format(loss))

        bad_by_scene = {"2": bad}
        loss2 = scorer.loss_relative(bad_by_scene, gt_by_scene)
        print("Loss2 of bad is : {}".format(loss2))

        # combined
        gt_by_scene = {"1": ground_truth, "2": ground_truth}
        comb_by_scene = {"1": good, "2": bad}
        loss_combined = scorer.loss_relative(comb_by_scene, gt_by_scene)
        print("Loss3, combined, is : {}".format(loss_combined))

    def test_loss_scene_absolute(self):
        try:
            scorer = MCSScore()

            # y_true = np.array([0, 0, 1, 1])
            # y_scores = np.array([0.1, 0.4, 0.35, 0.8])

            y_true = np.array([0, 0, 1, 1])
            y_scores = np.array([0.1, 0.4, 0.35, 0.8])

            loss_val = 1.0 - roc_auc_score(y_true, y_scores)
            print("Loss val: {}".format(loss_val))

            ground_truth = {'1': 0, '2': 1, '3': 1, '4': 0}
            good = {'1': 0.1, '2': 0.4, '3': 0.35, '4': 0.8}
            # good = {'1': 0.2, '2': 0.6, '3': 0.7, '4': 0.3}

            # get list of sorted keys
            good_sorted = []
            gt_sorted = []
            for key in sorted(ground_truth.keys()):
                gt_sorted.append(ground_truth[key])
                good_sorted.append(good[key])

            loss = scorer.loss_absolute(good_sorted, gt_sorted)
            print("Loss of good is : {}".format(loss))

            # bad = {'1': 0.9, '2': 0.1, '3': 0.3, '4': 0.8}
            # loss = scorer.loss_absolute(bad, ground_truth)
            # print("Loss of bad is : {}".format(loss))
        except Exception as err:
            print('print_exc():')
            traceback.print_exc(file=sys.stdout)
            print('print_exc(1):')
            traceback.print_exc(limit=1, file=sys.stdout)

    def test_score(self):
        try:
            scorer = MCSScore()

            sub_scene1 = {'1': 0.2, '2': 0.91, '3': 0.4, '4': 0.2}
            ref_scene1 = {'1': 1., '2': 1., '3': 0., '4': 0.}
            sub_scene = {"1": sub_scene1}
            ref_scene = {"1": ref_scene1}
            print("Result:  {}".format(scorer.loss_absolute(sub_scene, ref_scene)))
            print("Result:  {}".format(scorer.loss_relative(sub_scene, ref_scene)))
        except Exception as err:
            print('print_exc():')
            traceback.print_exc(file=sys.stdout)
            print('print_exc(1):')
            traceback.print_exc(limit=1, file=sys.stdout)
