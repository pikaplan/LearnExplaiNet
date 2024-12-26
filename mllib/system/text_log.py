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
from datetime import datetime


class TextLog(object):
  # --------------------------------------------------------------------------------------------------------------------
  def __init__(self, file_name, auto_flush):
    self.auto_flush = auto_flush
    self.last_flush = datetime.now()
    sMode = "w"
    if os.path.exists(file_name):
      sMode = "a"
    self.file = open(file_name, sMode)
    self.isclosed = False
  # --------------------------------------------------------------------------------------------------------------------
  def print(self, message):
    self.write(message + "\n")
  # --------------------------------------------------------------------------------------------------------------------
  def write(self, message):
    self.file.write(message)
    dNow = datetime.now()
    oDelta = dNow - self.last_flush
    if (oDelta.total_seconds() >= 5) or self.auto_flush:
      self.flush()
      self.last_flush = dNow
  # --------------------------------------------------------------------------------------------------------------------
  def flush(self):
    if not self.isclosed:
      self.file.flush()
  # --------------------------------------------------------------------------------------------------------------------
  def close(self):
    self.isclosed = True
    self.file.close()
  # --------------------------------------------------------------------------------------------------------------------
