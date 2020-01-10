import sys
import traceback
from unittest import TestCase
from mcs_score import MCSScore


class TestMCSScore(TestCase):

    def test_loss_scene_relative(self):
        scorer = MCSScore()

        ground_truth = {'1': 0, '2': 1, '3': 1, '4': 0}
        good = {'1': 0.2, '2': 0.6, '3': 0.7, '4': 0.3}

        loss = scorer.loss_scene_relative(good, ground_truth)
        print("Loss of good is : {}".format(loss))

        bad = {'1': 0.9, '2': 0.1, '3': 0.3, '4': 0.8}
        loss = scorer.loss_scene_relative(bad, ground_truth)
        print("Loss of bad is : {}".format(loss))

    def test_loss_scene_absolute(self):
        try:
            scorer = MCSScore()

            ground_truth = {'1': 0, '2': 1, '3': 1, '4': 0}
            good = {'1': 0.2, '2': 0.6, '3': 0.7, '4': 0.3}

            loss = scorer.loss_absolute(good, ground_truth)
            print("Loss of good is : {}".format(loss))

            bad = {'1': 0.9, '2': 0.1, '3': 0.3, '4': 0.8}
            loss = scorer.loss_absolute(bad, ground_truth)
            print("Loss of bad is : {}".format(loss))
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
