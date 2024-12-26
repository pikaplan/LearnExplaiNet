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
import glob
class FileObject(object):
  # ----------------------------------------------------------------------------------
  def __init__(self, filename, parent_folder=None, error_template=None, default_file_extension=None):
    self.filename = filename
    self.parent_folder = parent_folder
    self.error_template = error_template
    self.default_file_extension = default_file_extension
    if self.default_file_extension is not None:
      if not self.default_file_extension.startswith("."):
        self.default_file_extension = "." + self.default_file_extension
    self.filename       = filename
    self.parent_folder  = parent_folder
    self.error_template = error_template
    self.default_file_extension = default_file_extension
    if self.default_file_extension is not None:
      if not self.default_file_extension.startswith("."):
        self.default_file_extension = "." + self.default_file_extension
  # ----------------------------------------------------------------------------------
  def _useFileName(self, filename):
    if filename is not None:
        self.filename = filename
    if self.parent_folder is not None:
      sFilename = os.path.join(self.parent_folder, self.filename)
    else:
      sFilename = self.filename
    return sFilename
  # ----------------------------------------------------------------------------------
  def list_files(self, file_matching_pattern=None, is_full_path=True, is_removing_extension=False, sort_filename_key=None):
    if file_matching_pattern is None:
      file_matching_pattern = self.default_file_extension

    if file_matching_pattern is None:
      file_matching_pattern = "*.*"
    elif not file_matching_pattern.startswith("*."):
      file_matching_pattern = "*." + file_matching_pattern
    elif file_matching_pattern.startswith("."):
      file_matching_pattern = "*" + file_matching_pattern

    sEntries = glob.glob1(self.parent_folder, file_matching_pattern)

    if is_removing_extension:
      oFileNamesOnly = []
      for sEntry in sEntries:
        sFileNameOnly, _ = os.path.splitext(sEntry)
        oFileNamesOnly.append(sFileNameOnly)
      sEntries = sorted(oFileNamesOnly, key=sort_filename_key)

    if is_full_path:
      oResult = [os.path.join(self.parent_folder, x) for x in sEntries]
    else:
      oResult = [x for x in sEntries]

    return oResult
  # --------------------------------------------------------------------------------------------------------