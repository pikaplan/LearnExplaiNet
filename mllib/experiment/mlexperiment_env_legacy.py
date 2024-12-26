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


import shutil
import sys
from .MLExperimentConfig import CMLExperimentConfig
from mllib.system import CFileSystem
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
  if "Experiment.BaseName" in p_oDict :
    sBaseName = p_oDict["Experiment.BaseName"]
    nNumber   = int(p_oDict["Experiment.Number"])
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


  

  
  
  
            

# =========================================================================================================================
class CMLExperimentEnv(dict):
  # --------------------------------------------------------------------------------------
  def __init__(self, p_oFS, p_sBaseName, p_nNumber=None, p_sVariation=None, p_nFoldNumber=None):
    
    oConfigFS = p_oFS
    oModelFS  = p_oFS
    if isinstance(p_oFS, CFileSystem):
      oConfigFS = p_oFS.Configs
      oModelFS  = p_oFS.Models
    # ...................... | Fields | ......................
    self.ConfigFS = oConfigFS
    self.ModelFS  = oModelFS
    
    if p_nNumber is None:
      p_nNumber, p_sVariation = experimentNumberAndVariation(p_sVariation)

    self.BaseName     = p_sBaseName
    self.Number       = p_nNumber
    self.Variation    = p_sVariation
    self.FoldNumber   = p_nFoldNumber
    self.IsDebugable  = False
    self.IsRetraining = False
    
    self.Config = CMLExperimentConfig(self.ConfigFS.File(self.ExperimentCode + ".json"), p_nExperimentNumber=self.Number)
    self.ExperimentFS = self.ModelFS.SubFS(self.ExperimentCode)
    # ........................................................

  # --------------------------------------------------------------------------------------
  def copy_config(self, start_timestamp):
    sOriginalFileName = self.ConfigFS.File(f"{self.ExperimentCode}.json")
    sNewFileName = self.ExperimentFS.File(f"{start_timestamp}_{self.ExperimentCode}.json")
    shutil.copy(sOriginalFileName, sNewFileName)
  # --------------------------------------------------------------------------------------
  def move_log(self, start_timestamp, original_filename):
    self.copy_config(start_timestamp)
    sNewFileName = self.ExperimentFS.file(f"{start_timestamp}_{self.ExperimentCode}.{original_filename}")
    shutil.move(original_filename, sNewFileName)
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
    self.Number       = p_oParamsDict["ModelNumber"]
    self.IsDebugable  = p_oParamsDict["IsDebuggable"]
    self.IsRetraining = p_oParamsDict["IsRetraining"] 
    
    self.Config = CMLExperimentConfig(self.ModelFS.File(self.ExperimentCode + ".json"), p_nExperimentNumber=self.Number) 
  # --------------------------------------------------------------------------------------
# =========================================================================================================================


