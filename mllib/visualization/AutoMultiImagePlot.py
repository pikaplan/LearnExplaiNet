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

import numpy as np
import matplotlib.pyplot as plt


class CAutoMultiImagePlot(object):
  def __init__(self, min=None, max=None):
    self.Rows    = [] 
    self.RowCount = 0
    self.RowTitles = []    
    self.CurrentRow = -1
    self.RowColCount = dict()
    self.MaxColCount = 0
    self.min = min
    self.max = max
    
  def AddRow(self, p_sRowTitle=None):
    self.CurrentRow = self.RowCount
    self.Rows.append([])
    self.RowCount = len(self.Rows)
    self.RowTitles.append(p_sRowTitle) 
    
  def AddColumn(self, p_nImage, p_sTitle=None, p_sColorMap=None, p_sAspect=None,p_sExtent=None):
    oRowColumns = self.Rows[self.CurrentRow]
    dImage = { "image": p_nImage, "title": p_sTitle
              ,"cmap": p_sColorMap, "aspect": p_sAspect
              ,"extend": p_sExtent}
    
    oRowColumns.append(dImage)
    self.Rows[self.CurrentRow] = oRowColumns
    nColCount = len(oRowColumns)
    
    
    self.RowColCount[self.CurrentRow] = nColCount
    if nColCount > self.MaxColCount:
      self.MaxColCount = nColCount
  
  def Show(self, p_sTitle=None, p_nFigureSize=(15, 6), p_nRestrictColumns=None):
    
    nColumns = p_nRestrictColumns
    if nColumns is None:
      nColumns =self.MaxColCount
    fig, oSubplotGrid = plt.subplots(nrows=self.RowCount, ncols=nColumns
                                    , figsize=p_nFigureSize
                                    , subplot_kw={'xticks': [], 'yticks': []})
    plt.title(p_sTitle)
    bIsSingleRow = self.RowCount == 1
    if bIsSingleRow:
      oSubplotGrid = oSubplotGrid[np.newaxis, ...]
          
    for nRowIndex,oRowColumns in enumerate(self.Rows):
      if len(oRowColumns) > 0:
        sRowTitle = self.RowTitles[nRowIndex]
        nImageCount = len(oRowColumns)
        nIncr = nImageCount // nColumns
        nImageIndex = 0
        for nColIndex in range(nColumns):
          bMustPlot = True
          if (nIncr == 0) and (nColIndex > 0):
            bMustPlot = False
            
          if bMustPlot:
            dImage = oRowColumns[nImageIndex]          
            oSubPlot = oSubplotGrid[nRowIndex, nColIndex]
            oSubPlot.title.set_text(dImage["title"])
            oSubPlot.imshow(dImage["image"], cmap=dImage["cmap"],
                            aspect=dImage["aspect"], extent=dImage["extend"],
                            vmin=self.min, vmax=self.max
                            )
            if nColIndex == 0:
              oSubPlot.text(0.0, 0.5, sRowTitle, transform=oSubPlot.transAxes,
                      horizontalalignment='right', verticalalignment='center',
                      fontsize=9, fontweight='bold')          
          nImageIndex += nIncr  

    plt.show()
      
