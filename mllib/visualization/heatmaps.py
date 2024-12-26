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
import cv2

def plot_superimposed_heatmap(image, blend_heatmap, superimposed_heatmap, has_grid_lines=True, grid_line_color="gray"):
  # show images
  oFigure, oAxis = plt.subplots(2, 2)
  oAxis[0, 0].imshow(image, vmin=0, vmax=255)
  oAxis[0, 1].imshow(blend_heatmap, alpha=1., interpolation='gaussian', cmap=plt.cm.jet, vmin=0.0, vmax=1.0)
  oAxis[1, 0].imshow(superimposed_heatmap, alpha=1., interpolation='gaussian')#, cmap=plt.cm.jet)
  oAxis[1, 1].remove()

  major_ticks = np.arange(0, 28, 5)
  minor_ticks = np.arange(0, 28, 1)

  oTargetAxes = [oAxis[0,0], oAxis[1,0] ]
  if has_grid_lines:
    for oAx in oTargetAxes:
      oAx.set_xticks(major_ticks)
      oAx.set_xticks(minor_ticks, minor=True)
      oAx.set_yticks(major_ticks)
      oAx.set_yticks(minor_ticks, minor=True)
      oAx.grid(True, which="both", color=grid_line_color, linestyle='-', linewidth=0.5)
  plt.show()


def superimpose_heatmap(image, heatmap, is_gray_scale=True, alpha_blending=0.5, are_zeros_black=False):
  image = np.squeeze(image)
  if is_gray_scale:
    nImageHeight, nImageWidth = image.shape
    nImage = np.stack((image, image, image), axis=-1)
  else:
    nImageHeight, nImageWidth, _ = image.shape
    nImage = image

  # Replace zeros with black color

  nHeatmap = np.squeeze(heatmap.copy())
  nHeatmap = cv2.resize(nHeatmap, (nImageWidth, nImageHeight))

  cmap_jet = plt.cm.jet
  if are_zeros_black:
    cmap_jet.set_bad(color='black')
    nHeatmap[nHeatmap==0] = np.nan

  # Superimpose according to alpha blending
  nBlendHeatmap = cmap_jet(nHeatmap.astype(np.float32))
  nImageWithHeatmap = nBlendHeatmap[:, :, :3] * 255.0 * alpha_blending + nImage * (1.0 - alpha_blending)
  nImageWithHeatmap = nImageWithHeatmap.astype(np.uint8)

  return nImageWithHeatmap, nBlendHeatmap

