# ......................................................................................
# MIT License

# Copyright (c) 2024 Pantelis I. Kaplanoglou

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ......................................................................................


import numpy as np

from mllib.data import CDataSetBase
from keras.datasets import fashion_mnist
from mllib import FileStore

class TFDataSetBase(CDataSetBase):
  # --------------------------------------------------------------------------------------
  def __init__(self, datasets_folder, dataset_name, variation=""):
    super(TFDataSetBase, self).__init__(p_sName=dataset_name)

    self.Variation    = variation
    self.ClassCount   = 10
    self.class_names  = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    self.fs = FileStore(datasets_folder)
    self.fs.subfs(self.Name + self.Variation)
  # --------------------------------------------------------------------------------------
  def load(self):
    self.prepare()
  # --------------------------------------------------------------------------------------
  def prepare(self, is_overwriting=False):
    (oTSImages, oTSLabels), (oVSImages, oVSLabels) = fashion_mnist.load_data()#data_dir=self.fs.subfs("downloaded").base_folder)
    self.TrainingSet(oTSImages[...,np.newaxis], oTSLabels)
    self.ValidationSet(oVSImages[...,np.newaxis], oVSLabels)
  # --------------------------------------------------------------------------------------