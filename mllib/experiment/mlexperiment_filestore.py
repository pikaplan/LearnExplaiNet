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
import re
from mllib.storage import Storage
from mllib.system.filestore import FileStore

class ExperimentFileStore(FileStore):
  # --------------------------------------------------------------------------------------------------------------------
  def __init__(self, base_folder, is_verbose=False):
    super(ExperimentFileStore, self).__init__(base_folder, is_verbose=is_verbose, must_exist=True)
    self.checkpoint_fs = self.subfs("checkpoints", must_exist=True)
    self.logs_fs = self.subfs("logs", must_exist=True)
    self.weights_fs = self.subfs("weights")

    _, self.experiment_folder_name = os.path.split(self.base_folder)

    # Root
    self.configuration_filename = None
    self.timing_info_filename = None
    self.evaluation_filename = None
    self.training_log_filename = None
    self.best_checkpoint_files = None
    self.experiment_file_timestamp = None

    # Logs
    self.has_learning_history = False
    self.has_timing_info = False
    # Saved states
    self.best_checkpoints = None
    self.has_last_checkpoint = False
    self.has_weights = None

    self.list_root()
    self.list_logs()
    self.list_saved_state_checkpoints()

  # --------------------------------------------------------------------------------------------------------------------
  # TODO: Add the configuration, the training log, the epoch stats, timing info, timestamp
  def keep_best_state_only(self, archive_dest_folder):
    #for dCheckPoint in self.best_checkpoints:

    oCheckpointFiles = []
    for sFileIndex, sFileData in self.best_checkpoint_files:
      oCheckpointFiles.append(sFileIndex)
      oCheckpointFiles.append(sFileData)
    oCheckpointFiles.append(self.checkpoint_fs.file("checkpoint"))

    if len(oCheckpointFiles) > 0:
      self.archive_fs = FileStore(archive_dest_folder)
      sZipfileName = self.archive_fs.file(f"states_{self.experiment_folder_name}.zip")
      Storage.compress_files(oCheckpointFiles, sZipfileName)
      self.archive_fs.obj.save(self.best_checkpoints, f"states_{self.experiment_folder_name}.index.pkl")
  # --------------------------------------------------------------------------------------------------------------------
  def list_root(self):
    # Get the latest experiment configuration
    self.configuration_filename = None
    self.timing_info_filename = None
    for sFile in self.json.files:
      if "timing_info" in sFile:
        self.timing_info_filename = sFile
      else:
        # Get the first JSON config file except the timing_info
        self.configuration_filename = sFile

    for sFile in self.text.list_files("*.txt"):
      self.training_log_filename = sFile

    print(self.configuration_filename)
    print(self.timing_info_filename)
    print(self.training_log_filename)
  # --------------------------------------------------------------------------------------------------------------------
  def list_logs(self):
    self.has_learning_history = self.logs_fs.exists("keras_learning_history.pkl")
    self.has_timing_info = self.logs_fs.exists("timing_info.pkl")

    print(self.has_learning_history, self.has_timing_info)

  # --------------------------------------------------------------------------------------------------------------------
  def split_filename(self, filename):
    match = re.match(r'^(.*?)(\d)', filename)
    if match:
      return match.groups()
    else:
      return None
  # --------------------------------------------------------------------------------------------------------------------
  def list_saved_state_checkpoints(self):
    self.best_checkpoints = []
    self.best_checkpoint_files = []

    for sFile in self.checkpoint_fs.text.list_files("*.index", is_full_path=False):
      sFilenameIndex = self.checkpoint_fs.file(sFile)
      sCheckPointNumber, _ = os.path.splitext(sFile)
      sFilenameData = self.checkpoint_fs.file(f"{sCheckPointNumber}.data-00000-of-00001")

      dCheckPoint = { "Number":  int(sCheckPointNumber), "Type": "tf_checkpoint",
                      "FileName": sFile ,"FileNameIndex": sFilenameIndex, "FileNameData": sFilenameData }

      self.best_checkpoints.append(dCheckPoint)
      self.best_checkpoint_files.append([sFilenameIndex, sFilenameData])

    self.has_last_checkpoint = self.exists_folder("state")
    self.has_weights = self.weights_fs.has_files

    if len(self.best_checkpoints) > 0:
      print("Last Checkpoint:", self.best_checkpoints[-1], self.has_last_checkpoint, self.has_weights )
  # --------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
  oModels = FileStore(r"C:\MLModels\CI-CIFAR10")
  IS_FOLD_MODE = True
  if IS_FOLD_MODE:
    sSubFolders = sorted(oModels.subfolders)
    oExperimentRootFilestores = [FileStore(sFolder, must_exist=True) for sFolder in sSubFolders]
  else:
    oExperimentRootFilestores = [oModels]

  while len(oExperimentRootFilestores) > 0:
    oRootFS = oExperimentRootFilestores.pop(0)
    if IS_FOLD_MODE:
      _, sModelGroup = os.path.split(oRootFS.base_folder)
    print(f"Model group {sModelGroup}")

    sExperimentFolders = sorted(oRootFS.subfolders)
    for sExperimentFolder in sExperimentFolders:
      print(f" |__ {sExperimentFolder}")
      oSubFS = ExperimentFileStore(sExperimentFolder)

