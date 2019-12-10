#
#  Determine the ground truth for evaluation
#
# for each test, there are 6 possible outcomes:
#
#   0011
#   0101
#   0110
#   1001
#   1010
#   1100
#
# Use
import datetime

from PIL import Image, ImageDraw
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Slider
import sys

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
import pyqtgraph.metaarray as metaarray

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QSizePolicy, QSlider, QSpacerItem, \
    QVBoxLayout, QWidget

from answer import Answer

red = (255, 1, 1)
green = (1, 255, 1)
white = (255, 255, 255)

berkeley = "Berkeley-m2-learned-answer.txt"
gt = "ground_truth.txt"

class ClickableImageItem(pg.ImageItem):
    sigMouseClick = QtCore.pyqtSignal(object)

    def mouseClickEvent(self, ev):
        print("Clicked")


class KeyPressWindow(QtGui.QMainWindow):
    sigKeyPress = QtCore.pyqtSignal(object)
    sigMouseClick = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_function(self, fn):
        self.fn = fn

    def keyPressEvent(self, ev):
        self.fn(ev)

    def mouseReleaseEvent(self, ev):
        print("Mouse clicked event in window")


class Slider(QWidget):
    def __init__(self, minimum, maximum, parent=None):
        super(Slider, self).__init__(parent=parent)
        self.verticalLayout = QHBoxLayout(self)
        self.label = QLabel(self)
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout = QHBoxLayout()
        spacerItem = QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.slider = QSlider(self)
        self.slider.setOrientation(Qt.Horizontal)
        self.horizontalLayout.addWidget(self.slider)
        spacerItem1 = QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.resize(self.sizeHint())

        self.minimum = minimum
        self.maximum = maximum
        self.slider.valueChanged.connect(self.setLabelValue)
        self.x = None
        self.setLabelValue(self.slider.value())

    def setCallback(self, fn):
        self.slider.valueChanged.connect(fn)

    def setLabelValue(self, value):
        self.x = self.minimum + (float(value) / (self.slider.maximum() - self.slider.minimum())) * (
                self.maximum - self.minimum)
        self.label.setText("{0:d}".format(int(self.x)))


class TruthingViewer:

    def __init__(self):

        # init
        self.test_num = 1
        self.dataDir = Path("/mnt/ssd/cdorman/data/mcs/intphys/test/O2")
        self.masks = []
        self.image_map = {}
        self.texts = []
        self.image_items = [None] * 4

        self.selected = []

        # Windowing
        self.win = KeyPressWindow()
        self.win.set_function(self.update_keypress)
        self.win.resize(1500, 400)

        self.view = pg.GraphicsLayoutWidget()
        self.view.setBackground('w')
        self.win.setCentralWidget(self.view)
        self.win.show()
        self.win.setWindowTitle('truthing')

        # self.answer = self.read_answers(berkeley)
        self.ground_truth = self.read_answers(gt)
        self.write_results()

    def read_answers(self, filename):
        try:
            with open(filename) as answer_file:
                answer = Answer()
                answer.parse_answer_file(answer_file)
                return answer
        except:
            print(" No such file {}".format(filename))
            answer = Answer()
            return answer

    def set_test_num(self, test_num):
        if test_num < 0 or test_num > 1080:
            print("going off edge of tests")
            return

        self.test_num = test_num
        self.test_num_string = str(self.test_num).zfill(4)
        self.read_images()

    def read_images(self):
        self.image_map.clear()
        for scene in range(0, 4):
            frame_map = {}
            for frame_num in range(1, 101):
                frame_num_string = str(frame_num).zfill(3)
                image_name = self.dataDir / self.test_num_string / str(scene + 1) / "scene" / (
                        "scene_" + frame_num_string + ".png")
                img_src = mpimg.imread(str(image_name))
                frame_map[frame_num] = img_src
            self.image_map[scene] = frame_map

    def update_keypress(self, event):

        sys.stdout.flush()

        if event.key() == 70:
            self.set_test_num(self.test_num + 1)
        elif event.key() == 66:
            self.set_test_num(self.test_num - 1)
        elif event.key() == 49:    # This is '1'
            self.selected.append(1)
        elif event.key() == 50:    # this is '2'
            self.selected.append(2)
        elif event.key() == 51:   # this is '3'
            self.selected.append(3)
        elif event.key() == 52:
            self.selected.append(4)
        else:
            print("key: {}".format(event.key()))

        # If both have been selected, write out and reset
        if len(self.selected) == 2:
            self.set_results()
            self.write_results()
            self.selected.clear()

    def set_results(self):
        vals = [ 1, 1, 1, 1]
        vals[self.selected[0]-1] = 0
        vals[self.selected[1]-1] = 0
        block = 'O2'
        test = str(self.test_num).zfill(4)
        self.ground_truth.set_vals(block, test, vals)

    def write_results(self):
        gt_name = str(gt + datetime.datetime.now().isoformat())
        self.ground_truth.write_answer_file(gt_name)

    def mouseMoved(self, ev):
        print(" mouse moved {}".format(ev))

    def set_up_view(self, test_num):

        self.test_num = test_num
        self.set_test_num(test_num)

        for scene in range(0, 4):
            image_name = self.dataDir / self.test_num_string / str(scene + 1) / "scene" / "scene_001.png"
            img_src = mpimg.imread(str(image_name))
            self.image_items[scene] = pg.ImageItem(img_src, axisOrder='row-major', border='w')

            vb = self.view.ci.addViewBox(row=0, col=scene)

            proxy = pg.SignalProxy(vb.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
            vb.invertY()
            vb.addItem(self.image_items[scene])

            # self.text_items[scene] = pg.TextItem("text", color='b')
            # vb = self.view.ci.addViewBox(row=1, col=scene)
            # vb.addItem(self.text_items[scene])

        self.create_slider()

    def clicked(self, event):
        print("clicked event {}".format(event))

    def create_slider(self):
        horizLayout = QHBoxLayout(self.view)
        s = Slider(0, 100)
        s.setCallback(self.update_slider)
        horizLayout.addWidget(s)

    def update_slider(self, val):
        frame_num = int(val)
        # self.set_test_num(frame_num)
        if frame_num > 100 or frame_num < 1:
            return

        for text in self.texts:
            text.set_visible(False)
        self.texts.clear()

        for scene in range(0, 4):
            img = self.image_map[scene][frame_num]
            self.image_items[scene].setImage(img)

            # val = self.answer['O2'][self.test_num_string][str(scene+1)]
            # self.text_items[scene].setText(str(val))


if __name__ == "__main__":
    app = QtGui.QApplication([])

    dc = TruthingViewer()
    dc.set_up_view(1)

    QtGui.QApplication.instance().exec_()

    dc.get_input()