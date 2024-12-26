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
import glob
import json
import sys
if (sys.version_info.major == 3) and (sys.version_info.minor <= 7):
  import pickle5 as pickle
else:
  import pickle


from mllib.system.files import JSONFile
from mllib.system.files import PickleFile
from mllib.system.files import TextFile
from mllib.system.files import PNGFile
from .files.csvfile import CSVFile
# =======================================================================================================================
class CFileStoreLegacy(object):
  # --------------------------------------------------------------------------------------------------------
  #[Refactored]
  def __init__(self, p_sBaseFolder, p_bIsVerbose=False):
    #.......................... |  Instance Attributes | ............................
    self.BaseFolder = p_sBaseFolder
    self.IsVerbose  = p_bIsVerbose
    #................................................................................
    if not os.path.exists(self.BaseFolder):
      os.makedirs(self.BaseFolder)
  # --------------------------------------------------------------------------------------------------------
  # [Refactored]
  @property
  def HasData(self):
    bResult = os.path.exists(self.BaseFolder)
    if bResult:
      oFiles = os.listdir(self.BaseFolder)
      nFileCount = len(oFiles)
      bResult = nFileCount > 0

    return bResult;
  # --------------------------------------------------------------------------------------------------------
  # [Refactored]
  def Exists(self, p_sFileName):
    sFullFilePath = os.path.join(self.BaseFolder, p_sFileName)
    return os.path.isfile(sFullFilePath)
  # --------------------------------------------------------------------------------------------------------
  # [Refactored]
  def SubFS(self, p_sSubFolderName):
    return CFileStore(self.SubPath(p_sSubFolderName))
  # --------------------------------------------------------------------------------------------------------
  # [Refactored]
  def Files(self, p_sFileMatchingPattern, p_bIsRemovingExtension=False, p_tSortFilenamesAs=None):
    sEntries = glob.glob1(self.BaseFolder, p_sFileMatchingPattern)
    if p_bIsRemovingExtension:
      oFileNamesOnly = []
      for sEntry in sEntries:
        sFileNameOnly, _ = os.path.splitext(sEntry)
        oFileNamesOnly.append(sFileNameOnly)
      sEntries = sorted(oFileNamesOnly, key=p_tSortFilenamesAs)
    
    oResult = [os.path.join(self.BaseFolder, x) for x in sEntries]
     
    return oResult  
  # --------------------------------------------------------------------------------------------------------
  # [Refactored]
  @property
  def DirectoryEntries(self):
    return os.listdir(self.BaseFolder)
  # --------------------------------------------------------------------------------------------------------
  # [Refactored]
  @property
  def IsEmpty(self):
    return (len(os.listdir(self.BaseFolder)) == 0)
  # --------------------------------------------------------------------------------------------------------
  # [Refactored]
  def SubPath(self, p_sSubPath):
    if os.path.sep == "\\":
      if p_sSubPath.find(os.path.sep) < 0:
        p_sSubPath = p_sSubPath.replace("/", "\\")
    
    return self.Folder(p_sSubPath)
  # --------------------------------------------------------------------------------------------------------
  # [Refactored]
  def Folder(self, p_sSubPath):
    sFolder = os.path.join(self.BaseFolder, p_sSubPath)    
    if not os.path.exists(sFolder):
      os.makedirs(sFolder)
      
    return sFolder    
  # --------------------------------------------------------------------------------------------------------
  # [Refactored]
  def File(self, p_sFileName, p_sFileExt=None):
    if p_sFileExt is not None:
      p_sFileName += p_sFileExt
    return os.path.join(self.BaseFolder, p_sFileName)    
  # --------------------------------------------------------------------------------------------------------

  # [Refactored]
  def Deserialize(self, p_sFileName, p_bIsPython2Format=False, p_sErrorTemplate=None):
    """
    Deserializes the data from a pickle file if it exists.
    Parameters
        p_sFileName        : Full path to the  python object file 
    Returns
        The object with its data or None when the file is not found.
    """
    oData=None
    if (self.BaseFolder is not None):
      p_sFileName = os.path.join(self.BaseFolder, p_sFileName)
       
    if os.path.isfile(p_sFileName):
      if self.IsVerbose :
        print("      {.} Loading data from %s" % p_sFileName)

      with open(p_sFileName, "rb") as oFile:
        if p_bIsPython2Format:
          oUnpickler = pickle._Unpickler(oFile)
          oUnpickler.encoding = 'latin1'
          oData = oUnpickler.load()
        else:
          oData = pickle.load(oFile)
        oFile.close()
    else:
      if p_sErrorTemplate is not None:
        raise Exception(p_sErrorTemplate % p_sFileName)
       
    return oData
  #----------------------------------------------------------------------------------
  # [Refactored]
  def WriteTextToFile(self, p_sFileName, p_oText):
    """
    Writes text to a file

    Parameters
        p_sFileName        : Full path to the text file
        p_sText            : Text to write
    """
    if (self.BaseFolder is not None):
      p_sFileName = os.path.join(self.BaseFolder, p_sFileName)
    
    if self.IsVerbose :
      print("  {.} Saving text to %s" % p_sFileName)

    if isinstance(p_oText, list):
      with open(p_sFileName, "w") as oFile:
        for sLine in p_oText:
          print(sLine, file=oFile)
        oFile.close()    
    else:
      with open(p_sFileName, "w") as oFile:
        print(p_oText, file=oFile)
        oFile.close()

    return True
  #----------------------------------------------------------------------------------
  # [Refactored]
  def Serialize(self, p_sFileName, p_oData, p_bIsOverwritting=False, p_sExtraDisplayLabel=None):
    """
    Serializes the data to a pickle file if it does not exists.
    Parameters
        p_sFileName        : Full path to the  python object file 
    Returns
        True if a new file was created
    """
    bResult=False
    
    if (self.BaseFolder is not None):
      p_sFileName = os.path.join(self.BaseFolder, p_sFileName)

    if p_bIsOverwritting:
      bMustContinue = True
    else:
      bMustContinue = not os.path.isfile(p_sFileName)
        
    if bMustContinue:
      if self.IsVerbose :
        if p_sExtraDisplayLabel is not None:
            print("  {%s} Saving data to %s" % (p_sExtraDisplayLabel, p_sFileName) )                    
        else:
            print("  {.} Saving data to %s" % p_sFileName)
      with open(p_sFileName, "wb") as oFile:
          pickle.dump(p_oData, oFile, pickle.HIGHEST_PROTOCOL)
          oFile.close()
      bResult=True
    else:
      if self.IsVerbose:
          if p_sExtraDisplayLabel is not None:
              print("  {%s} Not overwritting %s" % (p_sExtraDisplayLabel, p_sFileName) )                    
          else:
              print("  {.} Not overwritting %s" % p_sFileName)
                            
    return bResult
  #----------------------------------------------------------------------------------
  # [Refactored]
  def LoadJSON(self, p_sFileName, p_sErrorTemplate=None):
    if (self.BaseFolder is not None):
      p_sFileName = os.path.join(self.BaseFolder, p_sFileName)
          
    dResult = None
    if os.path.exists(p_sFileName):
      with open(p_sFileName) as oFile:
        sJSON = oFile.read()
      dResult = json.loads(sJSON)
    else:
      if p_sErrorTemplate is not None:
        raise Exception(p_sErrorTemplate % p_sFileName)
      
    return dResult  
  #----------------------------------------------------------------------------------

  # [Refactored]
  def SaveJSON(self, p_sFileName, p_oObject):
    if (self.BaseFolder is not None):
      p_sFileName = os.path.join(self.BaseFolder, p_sFileName)
          
    sJSON = json.dumps(p_oObject, default=lambda o: o.__dict__, sort_keys=True, indent=4)
    with open(p_sFileName, "w") as oFile:
      oFile.write(sJSON)
      oFile.close()
  #----------------------------------------------------------------------------------
  # [Refactored]
  def __repr__(self)->str:
    return self.BaseFolder
  #----------------------------------------------------------------------------------
  # [Refactored]
  def __str__(self)->str:
    return self.BaseFolder
  #----------------------------------------------------------------------------------
# =======================================================================================================================





# =======================================================================================================================
class FileStore(object):
  # --------------------------------------------------------------------------------------------------------
  def __init__(self, base_folder, is_verbose=False, must_exist=False):
    #.......................... |  Instance Attributes | ............................
    self.base_folder = base_folder
    if not os.path.exists(self.base_folder):
      if must_exist:
        raise Exception(f"File store folder {self.base_folder} does not exist.")
      else:
        os.makedirs(self.base_folder)

    self.is_verbose  = is_verbose
    self.json = JSONFile(None, parent_folder=self.base_folder)
    self.obj = PickleFile(None, parent_folder=self.base_folder)
    self.text = TextFile(None, parent_folder=self.base_folder)
    self.csv = CSVFile(None, parent_folder=self.base_folder)
    self.img = PNGFile(None, parent_folder=base_folder)
    self.donefs = None
    #................................................................................
  # --------------------------------------------------------------------------------------------------------
  @property
  def has_files(self):
    return not self.is_empty
  # --------------------------------------------------------------------------------------------------------
  @property
  def is_empty(self):
    bExists = not os.path.exists(self.base_folder)
    if not bExists:
      bExists = len(os.listdir(self.base_folder)) > 0
    return not bExists
  # --------------------------------------------------------------------------------------------------------
  # COMPATIBILITY
  def Exists(self, filename):
    return self.exists(filename)
  # --------------------------------------------------------------------------------------------------------
  def exists_folder(self, filename):
    sFullPath = os.path.join(self.base_folder, filename)
    return os.path.exists(sFullPath)
  # --------------------------------------------------------------------------------------------------------
  def exists(self, filename):
    sFullFilePath = os.path.join(self.base_folder, filename)
    return os.path.isfile(sFullFilePath)
  # --------------------------------------------------------------------------------------------------------
  # COMPATIBILITY
  def SubFS(self, subfolder_name):
    return self.subfs(subfolder_name)
  # --------------------------------------------------------------------------------------------------------
  def subfs(self, subfolder_name, must_exist=False):
    return FileStore(self.subpath(subfolder_name), must_exist=must_exist)
  # --------------------------------------------------------------------------------------------------------
  # COMPATIBILITY
  def SubPath(self, subfolder_name):
    return self.subpath(subfolder_name)
  # --------------------------------------------------------------------------------------------------------
  def subpath(self, subfolder_name):
    if os.path.sep == "\\":
      if subfolder_name.find(os.path.sep) < 0:
        subfolder_name = subfolder_name.replace("/", "\\")
    return self.folder(subfolder_name)
  # --------------------------------------------------------------------------------------------------------
  # COMPATIBILITY
  def Folder(self, folder_name):
    return self.folder(folder_name)
  # --------------------------------------------------------------------------------------------------------
  def folder(self, folder_name):
    sFolder = os.path.join(self.base_folder, folder_name)
    if not os.path.exists(sFolder):
      os.makedirs(sFolder)

    return sFolder
  # --------------------------------------------------------------------------------------------------------
  # COMPATIBILITY
  def File(self, file_name, file_ext=None):
    return self.file(file_name, file_ext)
  # --------------------------------------------------------------------------------------------------------
  def file(self, file_name, file_ext=None):
    if file_ext is not None:
      if file_ext.find(".") < 0:
        file_name += "." + file_ext
      else:
        file_name += file_ext
    return os.path.join(self.base_folder, file_name)
  # --------------------------------------------------------------------------------------------------------
  def entries(self):
    return os.listdir(self.base_folder)
  # --------------------------------------------------------------------------------------------------------
  def __repr__(self)->str:
    return self.base_folder
  #----------------------------------------------------------------------------------
  def __str__(self)->str:
    return self.base_folder
  #----------------------------------------------------------------------------------


  # --------------------------------------------------------------------------------------------------------
  # COMPATIBILITY
  @property
  def BaseFolder(self):
    return self.base_folder
  # --------------------------------------------------------------------------------------------------------
  # COMPATIBILITY
  @property
  def HasData(self):
    return not self.is_empty
  # --------------------------------------------------------------------------------------------------------
  # COMPATIBILITY
  @property
  def IsEmpty(self):
    return self.is_empty
  # --------------------------------------------------------------------------------------------------------
  # COMPATIBILITY
  def Files(self, p_sFileMatchingPattern, p_bIsRemovingExtension=False, p_tSortFilenamesAs=None):
    return self.files(p_sFileMatchingPattern, p_bIsRemovingExtension, p_tSortFilenamesAs)
  # --------------------------------------------------------------------------------------------------------
  def files(self, file_matching_pattern, is_full_path=True, is_removing_extension=False, sort_filename_key=None):
    sEntries = glob.glob1(self.BaseFolder, file_matching_pattern)
    if is_removing_extension:
      oFileNamesOnly = []
      for sEntry in sEntries:
        sFileNameOnly, _ = os.path.splitext(sEntry)
        oFileNamesOnly.append(sFileNameOnly)
      sEntries = sorted(oFileNamesOnly, key=sort_filename_key)

    if is_full_path:
      oResult = [os.path.join(self.BaseFolder, x) for x in sEntries]
    else:
      oResult = [x for x in sEntries]

    return oResult
  # --------------------------------------------------------------------------------------------------------
  @property
  def subfolders(self, is_full_path=True):
    sResult = []
    for sFolder in os.listdir(self.base_folder):
      sFullPath = os.path.join(self.base_folder, sFolder)
      if os.path.isdir(sFullPath):
        if is_full_path:
          sResult.append(sFullPath)
        else:
          sResult.append(sFolder)
    return sResult
  # --------------------------------------------------------------------------------------------------------
  def dequeue_file(self, file_matching_pattern, is_full_path=True, archive_folder_name=".done"):
    if self.donefs is None:
      self.donefs = self.subfs(archive_folder_name)
    oQueue = self.files(file_matching_pattern, is_full_path=False)

    if len(oQueue) == 0:
      sFileName = None
    else:
      sFileName = oQueue[0]
      shutil.move(os.path.join(self.base_folder, sFileName), os.path.join(self.donefs.base_folder, sFileName))

    if is_full_path and (sFileName is not None):
      sFileName = os.path.join(self.donefs.base_folder, sFileName)
    return sFileName
  # --------------------------------------------------------------------------------------------------------
  def purge_done(self):
    #//TODO: Remove from the current filestore all files that are moved into the .done filestore
    pass
  # --------------------------------------------------------------------------------------------------------
  #COMPATIBILITY
  @property
  def DirectoryEntries(self):
    return os.listdir(self.base_folder)
  # ----------------------------------------------------------------------------------
  #COMPATIBILITY
  def Deserialize(self, filename, is_python2_format=False, error_template=None):
    return self.obj.load(filename, is_python2_format=is_python2_format, error_template=error_template)
  # ----------------------------------------------------------------------------------
  #COMPATIBILITY
  def Serialize(self, p_sFileName, p_oData, p_bIsOverwritting=False, p_sExtraDisplayLabel=None):
    self.obj.save(p_oData, p_sFileName, is_overwriting=p_bIsOverwritting, extra_display_label=p_sExtraDisplayLabel)
  # ----------------------------------------------------------------------------------
  #COMPATIBILITY
  def WriteTextToFile(self, filename, text_obj):
    self.text.save(text_obj, filename)
# ======================================================================================================================



# ======================================================================================================================
class CFileStore(FileStore):
  def __init__(self, base_folder, is_verbose=False):
    super(CFileStore, self).__init__(base_folder, is_verbose=is_verbose)
# ======================================================================================================================



if __name__ == "__main__":
  oFS = CFileStore("T:\MLModels.Keep\REXPLAINET_MNIST_16.10\checkpoints")
  sFiles = oFS.Files("*.index", True, p_tSortFilenamesAs=int)
  print("\n".join(sFiles))
  
  if False:
    oFS = CFileStore("MLData")
    print(oFS.SubPath("test/test2"))
    print(oFS.Folder("subfolder"))
    print(oFS.File("test"))
    print(oFS.SubFS("CIFAR10").File("test"))


