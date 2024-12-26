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
from mllib.system.core import system_name
from mllib.system.filestore import FileStore
from mllib.system.files import JSONFile
from mllib.system.MLSystem import CMLSystem



# =======================================================================================================================
class FileSystem(object):
  # --------------------------------------------------------------------------------------------------------
  def __init__(self, config_folder=None, model_folder=None, dataset_folder=None, model_group=None, must_exist=False):
    # Checks for the existence of the file system configuration for this host
    sFilename = system_name() + ".fsys"
    if os.path.exists(sFilename):
      oFile = JSONFile(sFilename)
      dConfig = oFile.load()
      config_folder = dConfig["Configs"]
      model_folder = dConfig["Models"]
      dataset_folder = dConfig["Datasets"]
      if model_group is None:
        model_group = dConfig["ModelGroup"]

    if model_group is not None:
      config_folder   = os.path.join(config_folder, model_group)
      model_folder    = os.path.join(model_folder, model_group)
    # ...................... | Fields | ......................
    self.configs   = FileStore(config_folder, must_exist=must_exist)
    self.models    = FileStore(model_folder, must_exist=must_exist)
    self.datasets  = FileStore(dataset_folder, must_exist=must_exist)
    # ........................................................

  # --------------------------------------------------------------------------------------------------------
  def use_for_mlsystem(self):
    CMLSystem.CONFIGS_FOLDER = self.configs.base_folder
    CMLSystem.MODEL_FOLDER = self.models.base_folder
    CMLSystem.DATASET_FOLDER = self.datasets.base_folder
  # --------------------------------------------------------------------------------------------------------
  def __str__(self)->str:
    sResult = f"  ConfigsFolder: \"{self.Configs.BaseFolder}\",\n"
    sResult += f"  ModelsFolder: \"{self.Models.BaseFolder}\",\n"
    sResult += f"  DatasetsFolder: \"{self.Datasets.BaseFolder}\"\n"
    sResult = "{\n" + sResult + "}"
    return sResult
  # --------------------------------------------------------------------------------------------------------
  def __repr__(self)->str:
    return self.__str__()
  # --------------------------------------------------------------------------------------------------------
# =======================================================================================================================
