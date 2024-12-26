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
from .filestore import CFileStore

# =======================================================================================================================
class CFileSystem(object):
  # --------------------------------------------------------------------------------------------------------
  def __init__(self, p_sConfigFolder, p_sModelFolder, p_sDatasetFolder, p_sModelGroup=None):
    if p_sModelGroup is not None:
      p_sConfigFolder = os.path.join(p_sConfigFolder, p_sModelGroup)
      p_sModelFolder = os.path.join(p_sModelFolder, p_sModelGroup)
    # ...................... | Fields | ......................
    self.Configs = CFileStore(p_sConfigFolder)
    self.Models = CFileStore(p_sModelFolder)
    self.Datasets = CFileStore(p_sDatasetFolder)
    # ........................................................

  # --------------------------------------------------------------------------------------------------------
  def __str__(self) -> str:
    sResult = f"  ConfigsFolder: \"{self.Configs.BaseFolder}\",\n"
    sResult += f"  ModelsFolder: \"{self.Models.BaseFolder}\",\n"
    sResult += f"  DatasetsFolder: \"{self.Datasets.BaseFolder}\"\n"
    sResult = "{\n" + sResult + "}"
    return sResult

  # --------------------------------------------------------------------------------------------------------
  def __repr__(self) -> str:
    return self.__str__()
  # --------------------------------------------------------------------------------------------------------

# =======================================================================================================================


