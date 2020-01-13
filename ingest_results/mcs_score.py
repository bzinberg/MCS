#!/usr/bin/env python
#
# author: Mathieu Bernard <mathieu.a.bernard@inria.fr>

import argparse
import os
import tempfile
import traceback
import zipfile
from pathlib import Path
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve

"""Evaluation script for the Intuitive Physics Challenge

Execute the script as follow::

    ./score.py input_dir output_dir

The `input_dir` MUST HAVE the two subdirectories `res` and `ref` and
the files `res/answer.txt` and `ref/answer.txt`.

The `output_dir` MUST BE an existing directory. The file
`output_dir/scores.txt` will be created (or overwritted if existing;)

"""

""" 
Modified by Clark Dorman for Machine Common sense.  The math is the same, but our 
directory structure is different.
"""


class MCSScore:

    @staticmethod
    def loss_scene_relative(sub_scene, ref_scene):
        """Computes the relative error rate for a single test, consisting of 4 scenes
        Equation 1 of https://arxiv.org/pdf/1803.07616.pdf
        Passed in ref_scene like:   { '1':   1, '2':   0, '3':   1, '4':   0 }
        and sub_scene like:         { '1': 0.8, '2': 0.1, '3': 0.6, '4': 0.55 }
        This computes the sum of plausibility scores for all the possible scenes and compares
        it to the sum of the plausibility scores for all the impossible scenes.  If the sum of
        plausibility of the impossible is more than the sum of possible, then loss is 1; else 0
        """
        sum_of_possible, sum_of_impossible = 0, 0
        loss = 0
        for k in ('1', '2', '3', '4'):
            if ref_scene[k] == 1:  # possible movie
                sum_of_possible += sub_scene[k]
            else:  # impossible movie
                sum_of_impossible += sub_scene[k]

        if sum_of_possible < sum_of_impossible:
            loss = 1

        return loss

    def loss_relative(self, submitted, reference):
        """Computes average relative loss over many scenes.
        Data should look like:
        {  "1" :   { '1': 0.8, '2': 0.1, '3': 0.6, '4': 0.55 },
           "2" :   { '1': 0.8, '2': 0.1, '3': 0.6, '4': 0.55 },
           etc.
        }
        where the key is the test consisting of 4 scenes.
        It is assumed that there is a submitted for each reference
        """
        num_tests = len(submitted)

        # Make sure that there are some tests, to avoid a divide by zero.
        if num_tests == 0:
            print("Possible problem:  No tests")
            return 0

        score = 0
        for scene in reference.keys():
            if scene not in submitted:
                print("No test {} in submitted".format(scene))
                score += 1
            else:
                score += self.loss_scene_relative(submitted[scene], reference[scene])

        return float(score) / float(num_tests)

    @staticmethod
    def loss_absolute(submitted, reference):
        """Computes the absolute error rate
        Equation 2 of https://arxiv.org/pdf/1803.07616.pdf
        Uses https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
        Note that this is not very interesting for a single test
        """
        y_true = np.array(reference)
        y_score = np.array(submitted)

        # y_true = np.asarray([
        #     y for k, v in sorted(reference.items())
        #     for _, y in sorted(v.items())], dtype=np.float32)
        #
        # y_score = np.asarray([
        #     y for k, v in sorted(submitted.items())
        #     for _, y in sorted(v.items())], dtype=np.float32)

        return 1.0 - roc_auc_score(y_true, y_score)

    def score_per_block(self, submitted, reference):
        # sub_occluded = {k: v for k, v in submitted.items() if 'occluded' in k}
        # sub_visible = {k: v for k, v in submitted.items() if 'visible' in k}
        #
        # ref_occluded = {k: v for k, v in reference.items() if 'occluded' in k}
        # ref_visible = {k: v for k, v in reference.items() if 'visible' in k}

        return {
            # 'visible': {
            #     'relative': self.loss_relative(sub_visible, ref_visible),
            #     'absolute': self.loss_absolute(sub_visible, ref_visible),
            # },
            # 'occluded': {
            #     'relative': self.loss_relative(sub_occluded, ref_occluded),
            #     'absolute': self.loss_absolute(sub_occluded, ref_occluded),
            # },
            'all': {
                'relative': self.loss_relative(submitted, reference),
                'absolute': self.loss_absolute(submitted, reference),
            }}

    def score(self, submitted, reference):
        """Computes the evaluation scores

        The scores are computed for all scenes, visible scenes and
        occluded scenes. For each category the absolute and relative error
        rate are evaluated and returned as a dictionary.

        """
        assert sorted(submitted.keys()) == sorted(reference.keys())

        return {block: self.score_per_block(
            {k: v for k, v in submitted.items() if block in k},
            {k: v for k, v in reference.items() if block in k})
            for block in ('O1', 'O2', 'O3')}

    def load_answer(self, answer_file):
        """Returns the content of `path`/answer.txt as a dict

        Returns
        -------
        answer : dict
            The output dict is structured as follow::

                {scene: {1: p_1, 2: p_2, 3: p_3, 4: p_4}}

            where p_i is the plausibility score for the associated movie.

        Raises
        ------
        ValueError
            If `path`/answer.txt does not exist
        AssertionError
            If the answers file is badly formatted

        """
        answer = self.nested_dict(3, float)
        # answer = collections.defaultdict(dict)
        with answer_file.open() as answer_file:
            for line in answer_file:
                split_line = line.split()
                assert len(split_line) == 2

                plausibility = float(split_line[1])
                assert 0 <= plausibility <= 1

                # Line looks like:  O3/1076/2 1
                first_part = split_line[0]
                key = first_part.split('/')
                block = str(key[0])
                test = str(key[1])
                scene = str(key[2])
                # print("{} {} {} {}".format(block, test, scene, split_line[1]))
                answer[block][test][scene] = plausibility

            for v in answer.values():
                for w in v.values():
                    # print("Keys: {}".format(w.keys()))
                    assert sorted(w.keys()) == ['1', '2', '3', '4']

        return answer

    def build_html(self, submitted):
        header = '<!DOCTYPE html>\n<html>\n<body>\n\n<p>\n'
        footer = '</p></body></html>\n'

        html = ''
        for k, v in sorted(submitted.items()):
            for w, x in sorted(v.items()):
                html += '{}_{}: {}<br>\n'.format(k, w, x)

        return header + html + footer

    def create_html_file(self, submitted, reference, output_dir):
        # build the html page with detailed results
        html_file = os.path.join(output_dir, 'scores.html')
        with open(html_file, 'w') as fout:
            fout.write(self.build_html(submitted))

        # compute the scores
        scores = self.score(submitted, reference)

        # write the final scores.txt file
        scores_file = os.path.join(output_dir, 'scores.txt')
        with open(scores_file, 'w') as fout:
            for k, v in sorted(scores.items()):
                for w, x in v.items():
                    for y, z in x.items():
                        fout.write('{}_{}_{}: {}\n'.format(k, w, y, z))

    def get_answer_from_zip(self, zip_filename):
        try:
            # Extract the data to a temp dir
            temp_dir = tempfile.mkdtemp()
            my_zip = zipfile.ZipFile(zip_filename)
            my_zip.extractall(temp_dir)

            answer_filename = "answer.txt"
            dir_path = Path(temp_dir)
            answer_path = dir_path / answer_filename
            return self.load_answer(answer_path)
        except Exception as e:
            print("Unable to read zip or parse answer.txt")
            traceback.print_exc()

    def nested_dict(self, n, type):
        """ Create a multi dimensional dictionary of dimension n.
        See: https://stackoverflow.com/questions/29348345/declaring-a-multi-dimensional-dictionary-in-python/39819609
        """
        if n == 1:
            return defaultdict(type)
        else:
            return defaultdict(lambda: self.nested_dict(n - 1, type))


def main_mcs():
    scorer = MCSScore()

    # load the submitted and reference data
    submitted_answer = scorer.get_answer_from_zip("submission_ibm_linear.zip")
    reference = scorer.load_answer(Path("ground_truth.txt"))
    output_dir = "."
    scorer.create_html_file(submitted_answer, reference, output_dir)


def parse_arguments():
    """Parses command-line arguments"""
    parser = argparse.ArgumentParser(description='scoring program for the IntPhys challenge')
    parser.add_argument('input_dir', help='directory containing reference and submission data')
    parser.add_argument('output_dir', help='where the scores.txt file is written by the scoring program')
    return parser.parse_args()


if __name__ == '__main__':
    main_mcs()
