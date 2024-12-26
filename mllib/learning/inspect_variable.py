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



from tensorflow.keras.callbacks import Callback
class InspectVariable(Callback):

  def __init__(self, variable_name):
    super(InspectVariable, self).__init__()
    self.variable_name = variable_name

  def on_epoch_end(self, epoch, logs=None):
     #//TODO: Temp is custom to resblocks. Make it generic
    oVars, oValues = [], []
    for oBlock in self.model.ResidualBlocks:
      var = getattr(oBlock, self.variable_name)
      oVars.append(var)
      value    = var.numpy()
      oValues.append(value)
    print(f"Epoch {epoch + 1}: {oVars} = {oValues}")

