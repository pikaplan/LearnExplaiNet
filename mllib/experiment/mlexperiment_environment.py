# ......................................................................................
# MIT License

# Copyright (c) 2023-2024 Pantelis I. Kaplanoglou

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
import shutil
import sys
import re
from datetime import datetime

from .MLExperimentConfig import CMLExperimentConfig
from mllib.system import FileSystem, FileStore, now_iso
from mllib.system.PrintTee import CPrintTee


# --------------------------------------------------------------------------------------
def experimentNumberAndVariation(p_sCode):
  if type(p_sCode) == int:
    nNumber = int(p_sCode)
    sVariation = None
  else:
    sParts = p_sCode.split(".")
    nNumber = int(sParts[0])
    if len(sParts) > 1:
      sVariation = sParts[1]
    else:
      sVariation = None

  return nNumber, sVariation
# --------------------------------------------------------------------------------------
def model_code(p_oDict):
  if "Experiment.BaseName" in p_oDict:
    sBaseName = p_oDict["Experiment.BaseName"]
    nNumber = int(p_oDict["Experiment.Number"])
    sVariation = None
    if "Experiment.Variation" in p_oDict:
      sVariation = p_oDict["Experiment.Variation"]
    nFoldNumber = None
    if "Experiment.FoldNumber" in p_oDict:
      nFoldNumber = p_oDict["Experiment.FoldNumber"]

    sCode = "%s_%02d" % (sBaseName, nNumber)
    if sVariation is not None:
      sCode += ".%s" % str(sVariation)
    if nFoldNumber is not None:
      sCode += "-%02d" % int(nFoldNumber)

  elif "ModelName" in p_oDict:
    sCode = p_oDict["ModelName"]
    if "ModelVariation" in p_oDict:
      sCode += "_" + p_oDict["ModelVariation"]
    if "ExperimentNumber" in p_oDict:
      sCode = sCode + "_%02d" % p_oDict["ExperimentNumber"]

  return sCode
# --------------------------------------------------------------------------------------
def experiment_code_and_timestamp(filename):
  sName, _ = os.path.splitext(os.path.split(filename)[1])
  sParts = re.split(r"_", sName, 2)
  sISODate = f"{sParts[0]}T{sParts[1][0:2]}:{sParts[1][2:4]}:{sParts[1][4:6]}"
  sExperimentCode = sParts[2]
  dRunTimestamp = datetime.fromisoformat(sISODate)
  return sExperimentCode, dRunTimestamp
# --------------------------------------------------------------------------------------


# =========================================================================================================================
class MLExperimentEnv(dict):

  # --------------------------------------------------------------------------------------
  @classmethod
  def experiment_filename_split(cls, filename):
    sTryFileName, sTryExt = os.path.splitext(filename)
    bIsVariationAndFold = "-" in sTryExt
    if bIsVariationAndFold:
      #LREXPLAINET22_MNIST_64.1-01
      sMainParts = filename.split("-")
      assert len(sMainParts) == 2, "Wrong experiment filename"
      sFoldNumber, _ = os.path.splitext(sMainParts[1])
      sParts = sMainParts[0].split("_")
      sModelName = f"{sParts[0]}_{sParts[1]}"
      sModelVar = sParts[2]
    else:
      sFileNameOnly, _ = os.path.splitext(filename)
      sMainParts = sFileNameOnly.split("-")
      if len(sMainParts) > 1:
        sFoldNumber = sMainParts[1]
      else:
        sFoldNumber = None
      sParts = sMainParts[0].split("_")
      sModelName = f"{sParts[0]}_{sParts[1]}"
      sModelVar = sParts[2]

    return sModelName, sModelVar, sFoldNumber
  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  @classmethod
  def preload_config(cls, config_folder, experiment_group=None, experiment_base_name=None, experiment_variation=None, experiment_fold_number=None, experiment_filename=None):
    oPrintOutput = []
    oPrintOutput.append(f"[?] Experiment started at {now_iso()}")
    oPrintOutput.append(f" |__ {'model':<24}: {experiment_base_name}")

    dExperimentSpec = {"base_name": experiment_base_name, "variation": experiment_variation, "fold_number": experiment_fold_number}
    if experiment_filename is not None:
      # LREXPLAINET18_MNIST_08.1-01.json
      _, sFileNameFull = os.path.split(experiment_filename)
      experiment_base_name, experiment_variation, experiment_fold_number = cls.experiment_filename_split(sFileNameFull)
      experiment_fold_number = int(experiment_fold_number)
      dExperimentSpec = {"base_name": experiment_base_name, "variation": experiment_variation, "fold_number": experiment_fold_number}

    if experiment_group is not None:
      config_folder = os.path.join(config_folder, experiment_group)
    oConfigFS = FileStore(config_folder, must_exist=True)

    if "." in experiment_variation:
      sParts = experiment_variation.split(".")
      experiment_variation = f"{int(sParts[0]):02d}.{sParts[1]}"
    else:
      experiment_variation = f"{int(experiment_variation):02d}"
    sMessage = f" |__ {'variation':<24}: {experiment_variation}"

    if experiment_fold_number is not None:
      experiment_variation = f"{experiment_variation}-{experiment_fold_number:02d}"
      sMessage += f" fold: {experiment_fold_number}"
    oPrintOutput.append(sMessage)

    if experiment_filename is not None:
      sExperimentFileName = experiment_filename
    else:
      sExperimentFileName = oConfigFS.file(f"{experiment_base_name}_{experiment_variation}.json")
    oConfig = CMLExperimentConfig(sExperimentFileName,p_nExperimentNumber=experiment_variation)
    oPrintOutput.append(f" |__ {'configuration file':<24}: {sExperimentFileName}")

    return oConfig, oPrintOutput, dExperimentSpec
  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  # --------------------------------------------------------------------------------------
  def __init__(self, p_oFS, base_name=None, number=None, variation=None, fold_number=None,
               experiment_filename=None, experiment_code=None, experiment_config=None, model_filestore=None):

    oConfigFS = p_oFS
    oModelFS = p_oFS
    if isinstance(p_oFS, FileSystem):
      oConfigFS = p_oFS.configs
      oModelFS = p_oFS.models

    if model_filestore is not None:
      oModelFS = model_filestore
    # ...................... | Fields | ......................
    self.ConfigFS = oConfigFS
    self.ModelFS = oModelFS
    self.experiment_filename = experiment_filename
    self.experiment_code     = None
    if experiment_code is not None:
      self.experiment_code = experiment_code
    elif self.experiment_filename is not None:
      _, sFileNameFull = os.path.split(experiment_filename)
      self.experiment_code, _ = os.path.splitext(sFileNameFull)

    if self.experiment_code is not None:
      base_name, variation, sFoldNumber = MLExperimentEnv.experiment_filename_split(self.experiment_code)
      if sFoldNumber is not None:
        fold_number = int(sFoldNumber)

    if number is None:
      number, variation = experimentNumberAndVariation(variation)

    self.BaseName = base_name
    self.Number = number
    self.Variation = variation
    self.FoldNumber = fold_number
    self.IsDebugable = False
    self.IsRetraining = False

    if self.experiment_filename is None:
      self.experiment_filename = self.ConfigFS.file(f"{self.ExperimentCode}.json")

    self.Config = experiment_config
    if self.Config is None:
      self.Config = CMLExperimentConfig(self.experiment_filename, p_nExperimentNumber=self.Number)
    self.ExperimentFS = self.ModelFS.subfs(self.ExperimentCode)
    # ........................................................

  # --------------------------------------------------------------------------------------
  @property
  def config(self):
    return self.Config
  # --------------------------------------------------------------------------------------
  def copy_config(self, start_timestamp):
    sOriginalFileName = self.experiment_filename
    sNewFileName = self.ExperimentFS.file(f"{start_timestamp}_{self.ExperimentCode}.json")
    shutil.copy(sOriginalFileName, sNewFileName)
  # --------------------------------------------------------------------------------------
  def move_log(self, start_timestamp, original_filename):
    self.copy_config(start_timestamp)
    _, original_filename_only = os.path.split(original_filename)
    sNewFileName = f"{start_timestamp}_{self.ExperimentCode}.{original_filename_only}"
    shutil.move(original_filename, self.ExperimentFS.file(sNewFileName))
    sys.stdout = CPrintTee(self.ExperimentFS.file(sNewFileName))
  # --------------------------------------------------------------------------------------
  @property
  def ExperimentCode(self):
    sCode = "%s_%02d" % (self.BaseName, self.Number)
    if self.Variation is not None:
      sCode += ".%s" % str(self.Variation)
    if self.FoldNumber is not None:
      sCode += "-%02d" % self.FoldNumber

    return sCode
  # --------------------------------------------------------------------------------------
  def AssignSystemParams(self, p_oParamsDict):
    self.Number = p_oParamsDict["ModelNumber"]
    self.IsDebugable = p_oParamsDict["IsDebuggable"]
    self.IsRetraining = p_oParamsDict["IsRetraining"]

    self.Config = CMLExperimentConfig(self.ModelFS.file(self.ExperimentCode + ".json"), p_nExperimentNumber=self.Number)
    # --------------------------------------------------------------------------------------
# =========================================================================================================================


