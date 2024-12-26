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

import cv2



from .fileobject import FileObject

class PNGFile(FileObject):
  # ----------------------------------------------------------------------------------
  def __init__(self, filename, parent_folder=None, error_template=None):
    super(PNGFile, self).__init__(filename, parent_folder, error_template, "png")
  # ----------------------------------------------------------------------------------
  def load(self, filename=None):
    filename = self._useFileName(filename)
    nImage = cv2.imread(filename)
    nImageRGB = cv2.cvtColor(nImage, cv2.COLOR_BGR2RGB)
    return nImageRGB
  # ----------------------------------------------------------------------------------
  def save(self, image_ndarray, filename=None):
    filename = self._useFileName(filename)
    cv2.imwrite(filename, image_ndarray)
  # ----------------------------------------------------------------------------------