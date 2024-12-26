# ......................................................................................
# MIT License

# Copyright (c) 2021-2024 Pantelis I. Kaplanoglou

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
import zipfile
import tarfile
from urllib.request import urlretrieve



# =======================================================================================================================
class DatasetDownloaderCIFAR10(object):
  DOWNLOAD_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

  # --------------------------------------------------------------------------------------------------------
  def __init__(self, dataset_folder):
    self.dataset_folder = dataset_folder
    self.temp_folder    = os.path.join(self.dataset_folder, "tmp")
  # --------------------------------------------------------------------------------------------------------            
  def _downloadProgressCallBack(self, count, block_size, total_size):
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()        
  # --------------------------------------------------------------------------------------------------------
  def __ensureDataSetIsOnDisk(self):
    if not os.path.exists(self.dataset_folder):
      os.makedirs(self.dataset_folder)
    if not os.path.exists(self.temp_folder):
      os.makedirs(self.temp_folder)


    sSuffix = DatasetDownloaderCIFAR10.DOWNLOAD_URL.split('/')[-1]
    sArchiveFileName = os.path.join(self.temp_folder, sSuffix)
        
    if not os.path.isfile(sArchiveFileName):
      sFilePath, _ = urlretrieve(url=DatasetDownloaderCIFAR10.DOWNLOAD_URL, filename=sArchiveFileName, reporthook=self._downloadProgressCallBack)
      print()
      print("Download finished. Extracting files.")

            
    if sArchiveFileName.endswith(".zip"):
      zipfile.ZipFile(file=sArchiveFileName, mode="r").extractall(self.temp_folder)
    elif sArchiveFileName.endswith((".tar.gz", ".tgz")):
      tarfile.open(name=sArchiveFileName, mode="r:gz").extractall(self.temp_folder)
    print("Done.")

    sSourceFolder = os.path.join(self.temp_folder, "cifar-10-batches-py")
    oFileNames = os.listdir(sSourceFolder)
    
    for sFileName in oFileNames:
        shutil.move(os.path.join(sSourceFolder, sFileName), self.dataset_folder)

    os.remove(sArchiveFileName)
    os.rmdir(sSourceFolder)
    os.rmdir(self.temp_folder)
  # --------------------------------------------------------------------------------------------------------
  def download(self):
    if not os.path.isfile(os.path.join(self.dataset_folder, "test_batch")):
      self.__ensureDataSetIsOnDisk()
  # --------------------------------------------------------------------------------------------------------
# =======================================================================================================================


if __name__ == "__main__":
  oDataSet = DatasetDownloaderCIFAR10()
  oDataSet.download()

