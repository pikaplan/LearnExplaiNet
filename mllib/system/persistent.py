# ......................................................................................
# MIT License

# Copyright (c) 2018-2024 Pantelis I. Kaplanoglou

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


# =======================================================================================================================
class PersistentObject(object):
  # --------------------------------------------------------------------------------------------------------
  def __init__(self, json=None):
    if json is not None:
      if isinstance(json, dict):
        self.__dict__ = json
      else:
        self.__dict__ = json.loads(json)

  # --------------------------------------------------------------------------------------------------------
  @classmethod
  def from_json(cls, filename):
    oResult = None

    if os.path.exists(filename):
      with open(filename) as oFile:
        sJSON = oFile.read()
      dConfig = json.loads(sJSON)
      oResult = cls(dConfig)

    return oResult
  # --------------------------------------------------------------------------------------------------------
  def to_json(self):
    return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
  # --------------------------------------------------------------------------------------------------------




# =======================================================================================================================


# =======================================================================================================================
class ConfigArgs(PersistentObject):
  pass
# =======================================================================================================================
class Config(PersistentObject):
  pass
# =======================================================================================================================