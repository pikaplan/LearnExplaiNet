# ......................................................................................
# MIT License

# Copyright (c) 2020-2024 Pantelis I. Kaplanoglou

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

# =========================================================================================================================
class GEmbeddingsProjectorExport(object):
  # --------------------------------------------------------------------------------------
  def __init__(self, p_sExportFolderName):
    self.ExportFolderName = p_sExportFolderName
  # -------------------------------------------------------------------------------------- 
  def Save(self, p_dWordsToTokens, p_nEmbeddingWeights):
    with open(os.path.join(self.ExportFolderName,'vectors.tsv'), 'w', encoding='utf-8') as oVectorsFile:
      with open(os.path.join(self.ExportFolderName,'metadata.tsv'), 'w', encoding='utf-8') as oMetadataFile:
        for nIndex in p_dWordsToTokens.keys():
          nTokenID  =  p_dWordsToTokens[nIndex]
          nIndex = nIndex - 1
          
          nVector = p_nEmbeddingWeights[nIndex]
          oVectorsFile.write('\t'.join([str(x) for x in nVector]) + "\n")
          oMetadataFile.write(nTokenID + "\n")
  # --------------------------------------------------------------------------------------          
# =========================================================================================================================
