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
from tqdm import tqdm

import matplotlib.pyplot as plt
from mllib.visualization import superimpose_heatmap, plot_superimposed_heatmap, ColorMaps


class DiscreteImagePlot(object):
  # --------------------------------------------------------------------------------------------------------------------
  def __init__(self, image, discrete_count, title=None, color_map=ColorMaps.discrete_101()):
    self.image = image
    self.discrete_count = discrete_count
    self.title = title
    self.title_fontsize = 22
    self.min = 0
    self.max = self.discrete_count - 1
    self.color_map = color_map
  # --------------------------------------------------------------------------------------------------------------------
  def close(self):
    plt.close(self.figure)
    self.figure = None
    plt.clf()
  # --------------------------------------------------------------------------------------------------------------------
  def _imshow(self):
    self.figure = plt.figure(figsize=(10, 10), constrained_layout=True)
    self.figure.suptitle(self.title, fontsize=str(self.title_fontsize))
    plt.imshow(self.image, cmap=ColorMaps.discrete_101(),
               interpolation="none", vmin=self.min, vmax=self.max)

    for i in range(self.image.shape[0]):
      for j in range(self.image.shape[1]):
        plt.text(j, i, f'{int(self.image[i, j])}', ha='center', va='center', color='white')
  # --------------------------------------------------------------------------------------------------------------------
  def show(self):
    self._imshow()
    plt.show()
    self.close()
    return self
  # --------------------------------------------------------------------------------------------------------------------
  def show_open(self):
    self._imshow()
    plt.show()
    self.close()
    return self
  # --------------------------------------------------------------------------------------------------------------------
  def save(self, filename):
    self._imshow()
    plt.savefig(filename)
    self.close()
    return self
  # --------------------------------------------------------------------------------------------------------------------




class MultiImageGrid(object):
  # --------------------------------------------------------------------------------------------------------------------
  def __init__(self, title, min_value=0.0, max_value=1.0, color_map=None,
               rows=None, columns=None, banner_image=None, title_fontsize=18):
    self.title = title
    self.min_value = min_value
    self.max_Value = max_value
    self.color_map = color_map
    self.rows = rows
    self.columns = columns
    self.title_fontsize = title_fontsize
    self.banner_image = banner_image

    self.figure = None
    self.grid_outer = None
    self.target_axes = []
  # --------------------------------------------------------------------------------------------------------------------
  def create_figure(self):
    self.figure = plt.figure(figsize=(self.columns * 2, (self.rows + 1) * 2), constrained_layout=True)
    self.grid_outer = self.figure.add_gridspec((self.rows + 1), self.columns, wspace=0, hspace=2)


    self.target_axes = []

    oGridInner = self.grid_outer[0, :].subgridspec(1, 1, wspace=0, hspace=0)
    oAxis = oGridInner.subplots()  # Create all subplots for the inner grid.
    oAxis.set(xticks=[], yticks=[])
    if self.banner_image is not None:
      oAxis.imshow(self.banner_image)
  # --------------------------------------------------------------------------------------------------------------------
  def load_images(self, image_grid, rows=None, columns=None, title=None):
    if title is None:
      title = self.title

    if (self.rows is None) or (self.columns is None):
      if isinstance(image_grid, np.ndarray):
        self.rows, self.columns = image_grid.shape[0:2]
      elif (rows is not None) and (columns is not None):
        self.rows, self.columns = rows, columns

    self.create_figure()
    self.figure.suptitle(title, fontsize=str(self.title_fontsize))

    nTotalCells = self.rows * self.columns
    with tqdm(desc=f"Creating plot", total=nTotalCells, position=0, leave=True) as oProgress:
      for nImageIndex in range(nTotalCells):
        nRowIndex = (nImageIndex // self.columns)
        nColIndex = (nImageIndex % self.columns)

        bIsHeatmap = False
        nImage, nHeatmap = None, None

        if isinstance(image_grid, np.ndarray):
          nImage = image_grid[nRowIndex, nColIndex]
        elif isinstance(image_grid, list):
          dImage = image_grid[nImageIndex]
          if isinstance(dImage, dict):
            nImage = dImage["image"]
            bIsHeatmap = "heatmap" in dImage
            if bIsHeatmap:
              nHeatmap = dImage["heatmap"]
          else:
            nImage = dImage

        nImageToPlot = nImage
        if bIsHeatmap:
          nImageToPlot, _ = superimpose_heatmap(nImage, nHeatmap, are_zeros_black=True)

        oGridInner = self.grid_outer[nRowIndex + 1, nColIndex].subgridspec(1, 1, wspace=0, hspace=0)
        oAxis = oGridInner.subplots()  # Create all subplots for the inner grid.
        oAxis.set(xticks=[], yticks=[])
        if bIsHeatmap:
          oAxis.imshow(nImageToPlot, alpha=1., interpolation='gaussian')
        else:
          oAxis.imshow(nImageToPlot, aspect='auto', vmin=self.min_value, vmax=self.max_Value,
                       interpolation='none', cmap=self.color_map)

          for i in range(nImageToPlot.shape[0]):
            for j in range(nImageToPlot.shape[1]):
              plt.text(j, i, f'{int(nImageToPlot[i, j])}', ha='center', va='center', color='white')

        self.target_axes.append(oAxis)
        oProgress.update(1)
  # --------------------------------------------------------------------------------------------------------------------
  def show(self, save_to_filename=None, tick_count=None):
    if tick_count is not None:
      major_ticks = np.arange(0, tick_count, 1)
      #minor_ticks = np.arange(0, ticks_count, 1)

    if tick_count is not None:
      for oAx in self.target_axes:
        oAx.set_xticks(major_ticks, minor=True)
        oAx.set_yticks(major_ticks, minor=True)
        oAx.grid(True, which="both", color="black", linestyle='dotted', linewidth=0.5)

    if save_to_filename is not None:
      plt.savefig(save_to_filename)
    else:
      plt.show()
    plt.close(self.figure)
    self.figure = None
    plt.clf()
  # --------------------------------------------------------------------------------------------------------------------