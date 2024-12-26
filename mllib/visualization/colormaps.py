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
import colorsys
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

class ColorMaps(object):
  _cmap_discrete_101 = None
  @classmethod
  def hsl_to_rgb(cls, hue, saturation, lightness):
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
    return red, green, blue

  @classmethod
  def discrete_101(cls, is_preview=False):
    if cls._cmap_discrete_101 is None:
      nLevels = 6
      # Black for zero
      oColors = [(0,0,0)]

      # Split into 12 hues, skip the last
      nHues = np.linspace(0, 1, 14, endpoint=False)
      nHues = nHues[:-2]

      # Split into 9 lightness levels, skip low and high
      nLightness = np.linspace(0.25, 0.80, num = 8, endpoint=False)

      # Constant saturation
      s = 0.6

      for h_index, h in enumerate(nHues):
          oLightness = nLightness
          if h_index ==len(nHues) - 1:
              oLightness = nLightness[-4:]
          for l_index, l in enumerate(oLightness):
              oColors.append(cls.hsl_to_rgb(h, s, l))
      for l in nLightness:
        oColors.append((l,l,l))

      #cls._cmap_discrete_101 = LinearSegmentedColormap.from_list('discrete100', oColors)
      cls._cmap_discrete_101 = ListedColormap(oColors,'discrete100')

    if is_preview:
      nData = np.linspace(0, 1, 100)
      nData = np.round(nData * 100)
      nData = nData.reshape(10, 10)
      # Display the colormap
      plt.imshow(nData, cmap=ColorMaps.discrete_101(), aspect='auto', interpolation="none")

      for i in range(nData.shape[0]):
        for j in range(nData.shape[1]):
          plt.text(j, i, f'{int(nData[i, j])}', ha='center', va='center', color='white')


      plt.axis('off')
      plt.show()

    return cls._cmap_discrete_101


if __name__ == '__main__':
  ColorMaps.discrete_101(True)



