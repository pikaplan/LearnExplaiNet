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

import os
import json
import glob
from .fileobject import FileObject

class JSONFile(FileObject):
  # ----------------------------------------------------------------------------------
  def __init__(self, filename, parent_folder=None, error_template=None):
    super(JSONFile, self).__init__(filename, parent_folder, error_template, "json")
  # ----------------------------------------------------------------------------------
  def load(self, filename=None):
    filename = self._useFileName(filename)

    dResult = None
    if os.path.exists(filename):
      with open(filename) as oFile:
        sJSON = oFile.read()
      dResult = json.loads(sJSON)
    else:
      if self.error_template is not None:
        raise Exception(self.error_template % filename)

    return dResult
  # ----------------------------------------------------------------------------------
  def save(self, obj, filename=None, is_sorted_keys=True):
    filename = self._useFileName(filename)

    if obj is not None:
      sJSON = json.dumps(obj, default=lambda o: o.__dict__, sort_keys=is_sorted_keys, indent=4)
      with open(filename, "w") as oFile:
        oFile.write(sJSON)
        oFile.close()
  # ----------------------------------------------------------------------------------
  @property
  def files(self, is_full_path=True):
    oResult = []
    if (self.parent_folder is not None):
      oJSONFiles = glob.glob(os.path.join(self.parent_folder, '*.json'))
      oJSONFiles = sorted(oJSONFiles, key=os.path.getmtime)

      for sJSONFile in oJSONFiles:
        if is_full_path:
          oResult.append(os.path.join(self.parent_folder, sJSONFile))
        else:
          oResult.append(sJSONFile)
    return oJSONFiles
  # ----------------------------------------------------------------------------------