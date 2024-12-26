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

import sys
import os
import random
from datetime import datetime
import numpy as np

from .filestore import CFileStore
from .filesystem_legacy import CFileSystem
from mllib.system.PrintTee import CPrintTee
import uuid

#==============================================================================================================================
class CMLSystem:
  IS_TENSORFLOW = True
  IS_TORCH      = False
  IS_TEE_PRINT  = True
  
  RANDOM_SEED     = 2000
  DATASET_FOLDER  = "MLData" 
  MODEL_FOLDER    = "MLModels"
  CONFIGS_FOLDER  = None        # set to "MLModels" for backwards compatibility
  IS_DETERMINISTIC          = False
  IS_FLOAT64                = False
  IS_USING_TENSORFLOAT32    = False

  IS_FFT_CONVOLUTION_ALGORITHM = False
  IS_AUTO_MIXED_PRECISION = False


  LOGS_FOLDER = ".mllogs"
  
  __instance = None
  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  @classmethod
  def IsTensorflow(cls):
      return cls.IS_TENSORFLOW
  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  @classmethod
  def Instance(cls, cmd_line_args=None):
    if cls.__instance is None:
      cls.__instance = CMLSystem(cmd_line_args)
      
    return cls.__instance

  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  @classmethod
  def prepare_environment(cls, is_fft_convolution_algorithm=False, is_auto_mixed_precision=False):
    oPrintOutput = []
    cls.IS_FFT_CONVOLUTION_ALGORITHM = is_fft_convolution_algorithm
    cls.IS_AUTO_MIXED_PRECISION = is_auto_mixed_precision

    if is_fft_convolution_algorithm:
      os.environ["TF_CUDNN_CONVOLUTION_BACKWARD_FILTER_ALGO"] = "fft"
      oPrintOutput.append("(~) Environment: cuDNN Backward Filter Algorithm is FFT")

    if is_auto_mixed_precision:
      os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
      oPrintOutput.append("(+) Environment: Allowing automixed precision")
    else:
      os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "0"
      oPrintOutput.append("(-) Environment: Disabled automixed precision")

    return oPrintOutput
  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



  #--------------------------------------------------------------------------------------------------------------
  def __init__(self, cmd_line_args):
    self.start_time = datetime.now()
    nProcGuid = uuid.uuid4()
    if not os.path.exists(CMLSystem.LOGS_FOLDER):
      os.makedirs(CMLSystem.LOGS_FOLDER)
    self._log_start_filename = os.path.join(CMLSystem.LOGS_FOLDER, f"log-{self.start_timestamp}-{str(nProcGuid)}.txt")
    if CMLSystem.IS_TEE_PRINT:
      self._keep_stdout = sys.stdout
      sys.stdout = CPrintTee(self._log_start_filename)

    # What tensorflow version is installed in this VM?
    if CMLSystem.IsTensorflow:
      import tensorflow as tf
      from .backends import TensorflowBackend
      TensorflowBackend.infrastructure_info()
      
    print("(>) Command line parameters:%s" % sys.argv)
  
    self.Params = dict()
    self.Params["ModelNumber"]  = None
    self.Params["IsDebuggable"] = False
    self.Params["IsRetraining"] = False
    
    bIsPythonNotebook = sys.argv[0].endswith("ipykernel_launcher.py")
    if not bIsPythonNotebook:
      if cmd_line_args is not None:
        if hasattr(cmd_line_args, "number_var"):
          self.Params["ModelNumber"] = cmd_line_args.number_var
      elif len(sys.argv) > 1:
        nModelNumber = 1
        oParam = sys.argv[1]
        if isinstance(oParam, int):
          nModelNumber = int(sys.argv[1])
        else:
          nModelNumber = oParam
        
        self.Params["ModelNumber"] = nModelNumber
      print("      |__ ModelNumber=%s" % str(self.Params["ModelNumber"]))

    nRandomSeed = CMLSystem.RANDOM_SEED
    if cmd_line_args is not None:
      if hasattr(cmd_line_args, "random_seed"):
        nRandomSeed = cmd_line_args.random_seed

    self.random_seed_all(nRandomSeed)
    if CMLSystem.IsTensorflow:
      if CMLSystem.IS_DETERMINISTIC:
        print("(+) Tensorflow: Enabling deterministic operations")
        tf.config.experimental.enable_op_determinism()
      if not CMLSystem.IS_USING_TENSORFLOAT32:
        print("(-) Tensorflow: Disabling tensor float 32")
        tf.config.experimental.enable_tensor_float_32_execution(False)
      if CMLSystem.IS_FLOAT64:
        print("(+) Tensorflow: Enabling float x64")
        tf.keras.backend.set_floatx('float64')
      
    self.ConfigFS   = None
    self.ModelFS    = None
    self.DatasetFS  = None
    if CMLSystem.CONFIGS_FOLDER is not None:
      self.ConfigFS   = CFileStore(CMLSystem.CONFIGS_FOLDER)
      self.ModelFS    = CFileStore(CMLSystem.MODEL_FOLDER)
      self.DatasetFS  = CFileStore(CMLSystem.DATASET_FOLDER)
    
    self.FileSys    = None
  # --------------------------------------------------------------------------------------------------------------
  @property
  def start_timestamp(self):
    return self.start_time.strftime("%Y-%m-%d_%H%M%S")
  # --------------------------------------------------------------------------------------------------------------
  @property
  def log_start_filename_to_move(self):
    sys.stdout.flush()
    sys.stdout = self._keep_stdout
    return self._log_start_filename
  # --------------------------------------------------------------------------------------
  def with_file_system(self, file_system):
    self.FileSys = file_system
    self.ConfigFS = self.FileSys.configs
    self.ModelFS = self.FileSys.models
    self.DatasetFS = self.FileSys.datasets

    CMLSystem.CONFIGS_FOLDER = self.ConfigFS.base_folder
    CMLSystem.MODEL_FOLDER = self.ModelFS.base_folder
    CMLSystem.DATASET_FOLDER = self.DatasetFS.base_folder

    return self
  # --------------------------------------------------------------------------------------
  def WithDefaultFileSystem(self):
    return self.WithFileSystem(CFileSystem("MLModels", "MLModels", "MLData"))
  # --------------------------------------------------------------------------------------
  def WithFileSystem(self, p_oFileSys):
    self.FileSys    = p_oFileSys
    self.ConfigFS   = self.FileSys.Configs
    self.ModelFS    = self.FileSys.Models
    self.DatasetFS  = self.FileSys.Datasets
    
    CMLSystem.CONFIGS_FOLDER  = self.ConfigFS.BaseFolder
    CMLSystem.MODEL_FOLDER    = self.ModelFS.BaseFolder
    CMLSystem.DATASET_FOLDER  = self.DatasetFS.BaseFolder
    
    return self
  # --------------------------------------------------------------------------------------
  # We are seeding the number generators to get some amount of determinism for the whole ML training process. 
  # This is not ensuring 100% deterministic reproduction of an experiment in GPUs
  def random_seed_all(self, p_nSeed):
    random.seed(p_nSeed)
    os.environ['PYTHONHASHSEED'] = str(p_nSeed)
    np.random.seed(p_nSeed)   
    if CMLSystem.IsTensorflow: 
      import tensorflow as tf
      tf.compat.v1.reset_default_graph()
      #tf.compat.v1.set_random_seed(cls.SEED)
      tf.random.set_seed(p_nSeed)
      tf.keras.utils.set_random_seed(p_nSeed)
  
    print("(>) Random seed set to %d" % p_nSeed)
  # --------------------------------------------------------------------------------------      
#==============================================================================================================================    
    
    
    
        
        
        
      