import os
import shutil
import zipfile
import pickle

# ......................................................................................
# MIT License

# Copyright (c) 2017-2024 Pantelis I. Kaplanoglou

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

# Part of original TALOS framework:
# ========================================================= ...
#
#        Tensor Abstraction Layer Objects 0.0.8-ALPHA
#        FILE SYSTEM OPERATIONS
#
#        Framework design by Pantelis I. Kaplanoglou
#
# ========================================================= ...


class Storage(object):

  # ------------------------------------------------------------------------------------
  @classmethod
  def join_paths(cls, paths):
    sResult = ""
    for sPath in paths:
      sResult = os.path.join(sResult, sPath)

    return sResult
  # ------------------------------------------------------------------------------------
  def create_flag_file(self, path, file_in_path=None, file_content=None):
    if file_in_path is not None:
      sFileName = os.path.join(path, file_in_path)
    else:
      sFileName = path

    with open(sFileName, "w") as oFile:
      if file_content is not None:
        print(file_content, file=oFile)
      else:
        print(".", file=oFile)
      oFile.close()
  # ------------------------------------------------------------------------------------
  @classmethod
  def split_filename(cls, path):
    path, filename_w_ext = os.path.split(path)
    filename, file_extension = os.path.splitext(filename_w_ext)
    return path, filename, file_extension
  # ------------------------------------------------------------------------------------
  @classmethod
  def join_filename(cls, path_to_folder, filename_only, extension):
    sFileName = os.path.join(path_to_folder, filename_only + extension)
    return sFileName
  # ------------------------------------------------------------------------------------
  @classmethod
  def split_last_folder(cls, path):
    return os.path.basename(os.path.normpath(path))
  # ------------------------------------------------------------------------------------
  @classmethod
  def ensure_folder(cls, path):
    """
    Ensures that p_sFolderName path exists

    Returns: True: if path already exists, False if path was not found and created
    """
    # TODO: Recursive for many dirs if python don't does this automatically
    Result = True
    if not os.path.exists(path):
      os.makedirs(path)
      Result = False

    return Result
  # ------------------------------------------------------------------------------------
  @classmethod
  def sanitize_filename(cls, filename):
    translation_table = str.maketrans('?:*/|"<>\\', "!.+-I'---")
    Result = filename.translate(translation_table)
    # print(p_sFileName, Result)
    return Result
  # ----------------------------------------------------------------------------------
  @classmethod
  def IsFolderEmpty(cls, p_sFolderName):
    return os.listdir(p_sFolderName) == []
  # ----------------------------------------------------------------------------------
  @classmethod
  def exists(cls, path, files=None):
    bResult = True
    if not os.path.exists(path):
      bResult = False

    if bResult and (files is not None):
      if isinstance(files, list):
        for sFileName in files:
          if not os.path.isfile(path + sFileName):
            bResult = False
            break
      else:
        bResult = os.path.isfile(os.path.join(path, files))
    else:
      os.path.isfile(path)

    return bResult
  # ----------------------------------------------------------------------------------
  @classmethod
  def ls_folders(cls, path):
    return sorted(next(os.walk(path))[1])
  # ----------------------------------------------------------------------------------
  @classmethod
  def ls(cls, path):
    return sorted(next(os.walk(path))[2])
  # ------------------------------------------------------------------------------------
  @classmethod
  def split_path(cls, path):
    sParentFolder = os.path.dirname(os.path.normpath(path))
    sSubFolder = os.path.basename(os.path.normpath(path))
    return sParentFolder, sSubFolder
  # ------------------------------------------------------------------------------------
  @classmethod
  def compress_files(cls, files, dest_zip_filename, is_verbose=False):
    oFileNames = []
    old_folder_name = None
    for sFileName in files:
      folder_name, file_name = cls.split_path(sFileName)
      oFileNames.append(file_name)
      if old_folder_name is not None:
        assert old_folder_name == folder_name, "File should be in the same folder"
      old_folder_name = folder_name

    sParentFolder, sFolder = cls.split_path(folder_name)
    sDestFolder, sFileName, sExt = cls.split_filename(dest_zip_filename)

    if is_verbose:
      print(f"  [.] Compressing files into [sZipFullPath] ...")
    bCanCreateZip = len(oFileNames) > 0
    if bCanCreateZip:
      with zipfile.ZipFile(dest_zip_filename, "w", zipfile.ZIP_DEFLATED) as oZip:
        for sFile in oFileNames:
          sSourceFileName = os.path.join(folder_name, sFile)
          sArchiveFileName = os.path.join(sFolder, sFile)
          if is_verbose:
            print("   |__ %s -> %s" % (sSourceFileName, sArchiveFileName))
          oZip.write(sSourceFileName, sArchiveFileName)
        oZip.close()

      if (sDestFolder is not None) and (sDestFolder.strip() != ""):
        cls.ensure_folder(sDestFolder)
        if is_verbose:
          print(f"  [.] Moving [{dest_zip_filename}] into folder sDestFolder ..." % (dest_zip_filename, sDestFolder))
        cls.move_file_to(dest_zip_filename, sDestFolder)
    else:
      dest_zip_filename = None

    return  dest_zip_filename
  # ------------------------------------------------------------------------------------
  @classmethod
  def compress_folder(cls, folder_name, dest_zip_filename, is_verbose=False):
    if not (cls.exists(folder_name)):
      return False, None

    sParentFolder, sSubFolder = cls.split_path(folder_name)

    if dest_zip_filename is not None:
      sDestFolder, sFileName, sExt = cls.split_filename(dest_zip_filename)
      sZipFileName = sFileName + sExt
    else:
      sDestFolder = None
      sZipFileName = sSubFolder + ".zip"

    sZipFullPath = os.path.join(sParentFolder, sZipFileName)

    sFiles = sorted(next(os.walk(folder_name))[2])
    if is_verbose:
      print("  {.} Compressing folder %s into [%s] ..." % (folder_name, sZipFullPath))
    bCanCreateZip = len(sFiles) > 0
    if bCanCreateZip:
      with zipfile.ZipFile(sZipFullPath, "w", zipfile.ZIP_DEFLATED) as oZip:
        for sFile in sFiles:
          sSourceFileName = os.path.join(folder_name, sFile)
          sArchiveFileName = os.path.join(sSubFolder, sFile)
          if is_verbose:
            print("   |__ %s -> %s" % (sSourceFileName, sArchiveFileName))
          oZip.write(sSourceFileName, sArchiveFileName)
        oZip.close()

      if (sDestFolder is not None) and (sDestFolder.strip() != ""):
        cls.ensure_folder(sDestFolder)
        if is_verbose:
          print("  {.} Moving [%s] into folder %s ..." % (sZipFullPath, sDestFolder))
        cls.move_file_to(sZipFullPath, sDestFolder)

    return bCanCreateZip, sZipFullPath
  # ------------------------------------------------------------------------------------
  @classmethod
  def decompress_file(cls, source_zip_files, dest_folder=None, is_verbose=False):
    bCanDecompress = os.path.isfile(source_zip_files)
    if bCanDecompress:
      if dest_folder is None:
        sDestFolder, _, _ = Storage.split_filename(source_zip_files)
      else:
        sDestFolder = dest_folder
      if is_verbose:
        print("  {.} Decompressing file [%s] into %s ..." % (source_zip_files, sDestFolder))
      with zipfile.ZipFile(source_zip_files, "r") as oZip:
        oZip.extractall(sDestFolder)
        oZip.close()
    return bCanDecompress

  # ------------------------------------------------------------------------------------
  @classmethod
  def delete_folder_files(cls, folder, is_verbose=True):
    if is_verbose:
      print("  {.} Deleting * in folder %s" % folder)
    sModelFiles = Storage.ls(folder)
    for sFile in sModelFiles:
      sFileNameFull = os.path.join(folder, sFile)
      if os.path.isfile(sFileNameFull):
        os.remove(sFileNameFull)
  # ------------------------------------------------------------------------------------
  @classmethod
  def delete_empty_folder(cls, folder):
    if os.path.exists(folder):
      bIsEmpty = os.listdir(folder) == []
      if bIsEmpty:
        shutil.rmtree(folder)
  # ------------------------------------------------------------------------------------
  @classmethod
  def delete_file(cls, filename):
    if os.path.isfile(filename):
      os.remove(filename)
  # ------------------------------------------------------------------------------------
  @classmethod
  def delete_folder(cls, folder, is_verbose=False):
    bMustRemove = os.path.exists(folder)
    if bMustRemove:
      if is_verbose:
        print("  {.} Removing folder %s" % folder)
      Storage.delete_folder_files(folder, is_verbose=False)
      os.removedirs(folder)
    else:
      if is_verbose:
        print("   -  Folder %s not found" % folder)
  # ----------------------------------------------------------------------------------
  @classmethod
  def copy_file(cls, p_sSourceFileName, p_sDestFileName, p_bIsOverwriting=False):
    if p_bIsOverwriting:
      if os.path.exists(p_sDestFileName):
        os.remove(p_sDestFileName)
      shutil.copyfile(p_sSourceFileName, p_sDestFileName)
    else:
      if not os.path.exists(p_sDestFileName):
        shutil.copyfile(p_sSourceFileName, p_sDestFileName)
  # ----------------------------------------------------------------------------------
  @classmethod
  def copy_file_to(cls, p_sSourceFileName, p_sDestFolder, p_bIsOverwriting=False):
    _, sFileNameWithExtension = os.path.split(p_sSourceFileName)
    Storage.copy_file(p_sSourceFileName, os.path.join(p_sDestFolder, sFileNameWithExtension),
                      p_bIsOverwriting=p_bIsOverwriting)

  # ----------------------------------------------------------------------------------
  @classmethod
  def move_file(cls, p_sSourceFileName, p_sDestFileName):
    if os.path.isfile(p_sSourceFileName):
      shutil.move(p_sSourceFileName, p_sDestFileName)
  # ----------------------------------------------------------------------------------
  @classmethod
  def move_file_to(cls, p_sSourceFileName, p_sDestFolder, p_sDestFileName=None):
    if os.path.isfile(p_sSourceFileName):
      if p_sDestFileName is not None:
        sFileNameWithExtension = p_sDestFileName
      else:
        _, sFileNameWithExtension = os.path.split(p_sSourceFileName)

      shutil.move(p_sSourceFileName, os.path.join(p_sDestFolder, sFileNameWithExtension))
  # ----------------------------------------------------------------------------------
  @classmethod
  def filename_version_suffix_next(cls, p_sFileName):
    """
    Filename versioning support: Gets the next version of the given filename
    """
    sNextVersionFileName = None
    for nIndex in range(0, 1000):
      sNextVersionFileName = p_sFileName[:-4] + "_r%03i" % nIndex + p_sFileName[-4:]
      if not os.path.isfile(sNextVersionFileName):
        break
    return sNextVersionFileName
  # ----------------------------------------------------------------------------------
  @classmethod
  def filename_version_suffix(cls, p_sFileName):
    """
    Filename versioning support: Gets the last version of the given filename
    """

    sLastVersionFileName = None
    if os.path.isfile(p_sFileName):
      sLastVersionFileName = p_sFileName
    else:
      for nIndex in range(0, 1000):
        sFileName = p_sFileName[:-4] + "_r%03i" % nIndex + p_sFileName[-4:]
        if not os.path.isfile(sFileName):
          break
        sLastVersionFileName = sFileName

    return sLastVersionFileName

  # ----------------------------------------------------------------------------------
  @classmethod
  def deserialize(cls, filename, is_versioned=False, is_python2_format=False, is_verbose=True):
    """
    Deserializes the data from a pickle file if it exists.
    Parameters
        p_sFileName        : Full path to the  python object file
    Returns
        The object with its data or None when the file is not found.
    """
    oData = None

    if is_versioned:
      filename = cls.filename_version_suffix(filename)

    if os.path.isfile(filename):
      if is_verbose:
        print("      {.} Loading data from %s" % filename)
      with open(filename, "rb") as oFile:
        if is_python2_format:
          oUnpickler = pickle._Unpickler(oFile)
          oUnpickler.encoding = 'latin1'
          oData = oUnpickler.load()
        else:
          oData = pickle.load(oFile)
        oFile.close()

    return oData

  # ----------------------------------------------------------------------------------
  @classmethod
  def serialize(cls, filename, object, is_overwritting=False, is_versioned=False, extra_display_label=None):
    """
    Serializes the data to a pickle file if it does not exists.
    Parameters
        p_sFileName        : Full path to the  python object file
    Returns
        True if a new file was created
    """
    bResult = False
    if is_versioned:
      bMustContinue = True
      filename = cls.filename_version_suffix_next(filename)
    else:
      if is_overwritting:
        bMustContinue = True
      else:
        bMustContinue = not os.path.isfile(filename)

    if bMustContinue:
      if extra_display_label is not None:
        print("  {%s} Saving data to %s" % (extra_display_label, filename))
      else:
        print("  {.} Saving data to %s" % filename)
      with open(filename, "wb") as oFile:
        pickle.dump(object, oFile, pickle.HIGHEST_PROTOCOL)
        oFile.close()
      bResult = True
    else:
      if extra_display_label is not None:
        print("  {%s} Not overwritting %s" % (extra_display_label, filename))
      else:
        print("  {.} Not overwritting %s" % filename)

    return bResult

  # ----------------------------------------------------------------------------------
  @classmethod
  def folder_size(cls, p_sFolder):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(p_sFolder):
      for f in filenames:
        fp = os.path.join(dirpath, f)
        total_size += os.path.getsize(fp)

    return total_size

  # ----------------------------------------------------------------------------------
  @classmethod
  def read_text_file(cls, p_sFileName):
    sResult = []
    if os.path.isfile(p_sFileName):
      with open(p_sFileName) as oInFile:
        oData = oInFile.readlines()

      for sLine in enumerate(oData):
        sResult.append(sLine[1].rstrip())

    return sResult

  # ----------------------------------------------------------------------------------
  @classmethod
  def current_folder(cls):
    return os.getcwd()
  # ----------------------------------------------------------------------------------

