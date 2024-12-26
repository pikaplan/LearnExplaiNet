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
import sys
import shutil

from .fileobject import FileObject
if (sys.version_info.major == 3) and (sys.version_info.minor <= 7):
  import pickle5 as pickle
else:
  import pickle

class PickleFile(FileObject):
  LOCAL_CACHE = None
  # ----------------------------------------------------------------------------------
  def __init__(self, filename, parent_folder=None, error_template=None, is_verbose=False):
    super(PickleFile, self).__init__(filename, parent_folder, error_template, "pkl")
    self.is_verbose = is_verbose
    self.has_local_cache = False #TODO
    self._cached_filename = None
    self._cached = None

    if (self.has_local_cache) and (not os.path.exists(PickleFile.LOCAL_CACHE)):
      os.makedirs(PickleFile.LOCAL_CACHE)
  # ----------------------------------------------------------------------------------
  def get(self, data_url):
    sParts = data_url.split("/")
    filename = sParts[0]
    dataprop = sParts[1]
    if self._cached_filename is None:
      self._cached_filename = filename
      self._cached = self.load(filename)
    return self._cached[dataprop]
  # ----------------------------------------------------------------------------------
  def close(self):
    self._cached_filename = None
    del self._cached
    self._cached = None
  # ----------------------------------------------------------------------------------
  def load(self, filename, is_python2_format=False, error_template=None):
    """
    Deserializes the data from a pickle file if it exists.
    Parameters
        p_sFileName        : Full path to the  python object file
    Returns
        The object with its data or None when the file is not found.
    """
    sFileName = self._useFileName(filename)
    oData = None

    if (self.has_local_cache):
      sOriginalFileName = sFileName
      sFileName = os.path.join(PickleFile.LOCAL_CACHE, sFileName.replace(os.path.sep, "_"))
      if os.path.isfile(sOriginalFileName):
        if not os.path.exists(sFileName):
          print("... Caching file [%s]" % sFileName)
          shutil.copyfile(sOriginalFileName, sFileName)

    if os.path.isfile(sFileName):
      if self.is_verbose:
        print("      {.} Loading data from %s" % sFileName)

      with open(sFileName, "rb") as oFile:
        if is_python2_format:
          oUnpickler = pickle._Unpickler(oFile)
          oUnpickler.encoding = 'latin1'
          oData = oUnpickler.load()
        else:
          oData = pickle.load(oFile)
        oFile.close()
    else:
      if error_template is not None:
        raise Exception(error_template % sFileName)

    return oData
  # ----------------------------------------------------------------------------------
  def save(self, obj, filename=None, is_overwriting=False, extra_display_label=None):
    """
    Serializes the data to a pickle file if it does not exists.
    Parameters
        p_sFileName        : Full path to the  python object file
    Returns
        True if a new file was created
    """
    bResult = False
    sFileName = self._useFileName(filename)

    if is_overwriting:
      bMustContinue = True
    else:
      bMustContinue = not os.path.isfile(sFileName)

    if bMustContinue:
      if self.is_verbose:
        if extra_display_label is not None:
          print("  {%s} Saving data to %s" % (extra_display_label, sFileName))
        else:
          print("  {.} Saving data to %s" % sFileName)
      with open(sFileName, "wb") as oFile:
        pickle.dump(obj, oFile, pickle.HIGHEST_PROTOCOL)
        oFile.close()
      bResult = True
    else:
      if self.is_verbose:
        if extra_display_label is not None:
          print("  {%s} Not overwritting %s" % (extra_display_label, sFileName))
        else:
          print("  {.} Not overwritting %s" % sFileName)

    return bResult
  # ----------------------------------------------------------------------------------

import platform
if platform.system() == "Windows":
  PickleFile.LOCAL_CACHE = "C:\\MLCache"
else:
  PickleFile.LOCAL_CACHE = "/MLCache"


