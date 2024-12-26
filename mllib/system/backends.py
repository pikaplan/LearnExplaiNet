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
import logging

#==============================================================================================================================
class TensorflowBackend:
  # --------------------------------------------------------------------------------------------------------------
  @classmethod
  def run_on_cpu(cls):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU only
    import tensorflow as tf
    cls.IS_IMPORTED = True
    # Configure TensorFlow to use CPU only
    tf.config.set_visible_devices([], 'GPU')

  # --------------------------------------------------------------------------------------------------------------
  @classmethod
  def infrastructure_info(cls, is_tree_like=True):
    dInfraSetup = cls.infrastructure_setup()
    if is_tree_like:
      bHasPrintedTensorflowNode = False
      print("[~] Machine Learning Compute Infrastructure")
      for sKey, oValue in dInfraSetup.items():
        if sKey.startswith("Tensorflow"):
          if not bHasPrintedTensorflowNode:
            bHasPrintedTensorflowNode = True
            print(f" |__ (T) Tensorflow")
          print(f"      |__ {sKey:<49}: {oValue}")
        else:
          print(f" |__ {sKey:<49}: {oValue}")
    else:
      print("="*100)
      sHeader = "Machine Learning Compute Infrastructure"
      print(f"|{sHeader:^98}|")
      print("-" * 100)
      for sKey, oValue in dInfraSetup.items():
        print(f"{sKey:<49}: {oValue}")
      print("-" * 100)
  # --------------------------------------------------------------------------------------------------------------
  @classmethod
  def infrastructure_setup(cls):
    import tensorflow as tf
    dSetup = dict()

    # Reads the GPU info
    dGPUInfo = cls.gpu_info()
    nGPUComputeCapability = dGPUInfo["compute_capability"]
    dSetup["GPU.Name"] = dGPUInfo["device_name"]
    dSetup["GPU.ComputeCapability"] = f"{nGPUComputeCapability[0]}.{nGPUComputeCapability[1]}"


    # Reads the CUDA version through nvcc and cuDNN version through platform specific code
    for nIndex, s in enumerate(cls.nvcc()):
      #print(f"{nIndex}:{s}")
      if s.lower().startswith("cuda compilation tools"):
        sParts = s.split(",")
        dSetup["CUDA.Release"] = sParts[1].strip()
        dSetup["CUDA.Version"] = sParts[2].strip()
    dSetup["CUDA.cuDNN.Version"] = cls.cudnn_version()

    dSetup["Tensorflow.Version"] = tf.__version__

    # Reads the Tensorflow build info
    dBuildInfo = cls.build_info()
    #for s in dBuildInfo.keys():
    #  print(s, dBuildInfo[s])
    dSetup["Tensorflow.BuildFor.CUDA"] = dBuildInfo["cuda_version"]
    dSetup["Tensorflow.BuildFor.cuDNN"] = dBuildInfo["cudnn_version"]
    dSetup["Tensorflow.BuildFor.SupportedComputeCapabilities"] = dBuildInfo["cuda_compute_capabilities"]

    nCurrentIndex = -1
    oSupportedCCs = []
    for s in dBuildInfo["cuda_compute_capabilities"]:
      sParts = s.split("_")
      oCC = (int(sParts[1][0]), int(sParts[1][1]))
      if (oCC[0] == nGPUComputeCapability[0]) and (oCC[1] == nGPUComputeCapability[1]):
        nCurrentIndex = len(oSupportedCCs)
      oSupportedCCs.append(f"{oCC[0]}.{oCC[1]}")

    dSetup["Tensorflow.BuildFor.SupportedComputeCapabilities"] = oSupportedCCs
    dSetup["Tensorflow.UsedComputeCapability"] = oSupportedCCs[nCurrentIndex]
    dSetup["Tensorflow.GPUs.Available"] = tf.config.list_physical_devices('GPU')
    dSetup["Tensorflow.GPUs.Visible"] = tf.config.get_visible_devices("GPU")
    dSetup["Tensorflow.GPUs"] = tf.test.gpu_device_name()
    return dSetup
  # --------------------------------------------------------------------------------------------------------------
  @classmethod
  def gpu_info(cls):
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    dResult = dict()
    if len(gpus) > 0:
      dDeviceDetails = tf.config.experimental.get_device_details(gpus[0])
      for sKey in dDeviceDetails.keys():
        dResult[sKey] = dDeviceDetails[sKey]

    return dResult
  # --------------------------------------------------------------------------------------------------------------
  @classmethod
  def run_on_gpu(cls, gpu_index):
    print(f"/i\ Executing on GPU #{gpu_index} ")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
  # --------------------------------------------------------------------------------------------------------------
  @classmethod
  def set_mixed_precision(cls, is_enabled):
    if is_enabled:
      print(f"/!\ Allowing automixed precision")
      os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
    else:
      os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "0"
  #--------------------------------------------------------------------------------------------------------------
  @classmethod
  def info(cls):
    import tensorflow as tf
    print("(T) Tensorflow version " + tf.__version__)
    print(" |__ GPUs Available      : ", tf.config.list_physical_devices('GPU'))
    print(" |__ Visible GPUs        : ", tf.config.get_visible_devices("GPU"))
    print(" |__ Default GPU Device  :%s" % tf.test.gpu_device_name())
  #--------------------------------------------------------------------------------------------------------------
  @classmethod
  def debug(cls, is_active=True):
    import tensorflow as tf
    tf.config.run_functions_eagerly(is_active)
  # --------------------------------------------------------------------------------------------------------------
  @classmethod
  def memory_growth(cls, is_incremental=True):
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, is_incremental)
  # --------------------------------------------------------------------------------------------------------------
  @classmethod
  def debug_autograph(cls):
    import tensorflow as tf
    tf.autograph.set_verbosity(3, True)
  #--------------------------------------------------------------------------------------------------------------
  @classmethod
  def build_info(cls):
    from tensorflow.python.platform import build_info as tf_build_info

    dBuildInfo = dict()
    for sKey in tf_build_info.build_info.keys():
      dBuildInfo[sKey] = tf_build_info.build_info[sKey]

    return dBuildInfo

  # --------------------------------------------------------------------------------------------------------------
  @classmethod
  def list_libcudnn(cls):
    import subprocess
    oOutput = subprocess.check_output("apt list --installed | grep libcudnn", shell=True)
    oOutputLines = oOutput.decode().splitlines()

    oResult = []
    for sLine in oOutputLines:
      oResult.append(sLine)

    return oResult
  # --------------------------------------------------------------------------------------------------------------
  @classmethod
  def locate_cudnn_dll(cls):
    sCUDA = cls.cuda_version()
    sCUDNN = cls.build_info()["cudnn_version"]
    return f"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v{sCUDA}\\bin\\cudnn{sCUDNN}.dll"
  # --------------------------------------------------------------------------------------------------------------
  @classmethod
  def _is_windows(cls):
    import platform
    sPlatform = platform.system()
    return (sPlatform == "Windows")
  # --------------------------------------------------------------------------------------------------------------
  @classmethod
  def _is_wsl(cls):
    try:
      with open('/proc/version', 'r') as f:
        version_info = f.read().lower()
        if 'microsoft' in version_info or 'wsl' in version_info:
          return True
    except FileNotFoundError:
      pass
    if 'WSL_INTEROP' in os.environ:
      return True
    if os.path.exists('/proc/sys/fs/binfmt_misc/WSLInterop'):
      return True
    return False
  # --------------------------------------------------------------------------------------------------------------
  @classmethod
  def _dll_info(cls, dll_filename):
    dInfo = dict()
    if cls._is_windows:
      import win32api
      dFileVersionsInfo = win32api.GetFileVersionInfo(dll_filename, '\\VarFileInfo\\Translation')
      ## \VarFileInfo\Translation returns list of available (language, codepage) pairs that can be used to retreive string info
      ## any other must be of the form \StringfileInfo\%04X%04X\parm_name, middle two are language/codepage pair returned from above
      dInfo = dict()
      for sLanguageCode, sCodePage in dFileVersionsInfo:
        dInfo["lang"] = sLanguageCode
        dInfo["codepage"] = sCodePage

        #print('lang: ', lang, 'codepage:', codepage)
        oVersionStrings = ('Comments', 'InternalName', 'ProductName',
                       'CompanyName', 'LegalCopyright', 'ProductVersion',
                       'FileDescription', 'LegalTrademarks', 'PrivateBuild',
                       'FileVersion', 'OriginalFilename', 'SpecialBuild')
        for sVersionString in oVersionStrings:
          str_info = u'\\StringFileInfo\\%04X%04X\\%s' % (sLanguageCode, sCodePage, sVersionString)
          dInfo[sVersionString] = repr(win32api.GetFileVersionInfo(dll_filename, str_info))
    return dInfo

  # --------------------------------------------------------------------------------------------------------------
  @classmethod
  def cudnn_version(cls):
    sVersion = "not installed"
    if cls._is_windows():
      sVersion = cls._dll_info(cls.locate_cudnn_dll())["ProductVersion"]
      sVersion = sVersion[1:-1].replace(",", ".")
    else:
      if cls._is_wsl():
        pass
      else:
        oListOfInstalledCuDNN = cls.list_libcudnn()
        for sItem in oListOfInstalledCuDNN:
          if sItem.startswith("libcudnn8/"):
            sParts = sItem.split("/")
            sParts = sParts[1].split(" ")
            sVersion = sParts[1]
    return sVersion
  # --------------------------------------------------------------------------------------------------------------
  @classmethod
  def cuda_version(cls):
    for nIndex, s in enumerate(cls.nvcc()):
      if s.lower().startswith("cuda compilation tools"):
        # Cuda compilation tools, release 11.2, V11.2.142
        sParts = s.split(",")
        for sPart in sParts:
          # V11.2.142
          if sPart.strip().upper().startswith("V"):
            # 11.2.142
            sParts = sPart.strip()[1:].split(".")
            sVersion = f"{sParts[0]}.{sParts[1]}"
            return sVersion
    return "not installed"
  # --------------------------------------------------------------------------------------------------------------
  @classmethod
  def nvcc(cls):
    import subprocess
    oResult = []
    if not cls._is_wsl():
      try:
        oOutput = subprocess.check_output("nvcc --version", shell=True)
        oOutputLines = oOutput.decode().splitlines()
        for sLine in oOutputLines:
          oResult.append(sLine)
      except:
        pass

    return oResult
  # --------------------------------------------------------------------------------------------------------------
  @classmethod
  def warnings(cls, are_enabled):
    import tensorflow as tf
    if are_enabled:
      tf.get_logger().setLevel(logging.WARNING)
    else:
      tf.get_logger().setLevel(logging.ERROR)
  #--------------------------------------------------------------------------------------------------------------


#==============================================================================================================================


if __name__ == "__main__":
  TensorflowBackend.infrastructure_info()