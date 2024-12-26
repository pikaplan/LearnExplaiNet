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

from keras import optimizers
from .LearningRateSchedule import CLearningRateSchedule

# =========================================================================================
class CLearningAlgorithm(object):
  # -----------------------------------------------------------------------------------
  def __init__(self, p_oConfig):
    self.Config = p_oConfig
    self.Optimizer = None
    self.Callbacks = []
    
    self.optimizer_name = self.Config["Training.Optimizer"].upper()
    if self.optimizer_name == "SGD":
      self.Optimizer = optimizers.SGD(learning_rate=self.Config["Training.LearningRate"], momentum=self.Config["Training.Momentum"])
      oLearningRateSchedule = CLearningRateSchedule(self.Config)
      self.Callbacks.append(oLearningRateSchedule)
    elif self.optimizer_name == "ADAM":
      self.Optimizer = optimizers.Adam(learning_rate=self.Config["Training.LearningRate"])
    elif self.optimizer_name == "RMSPROP":
      #//TODO: Rho
      if "Training.Momentum" in self.Config:
        self.Optimizer = optimizers.RMSprop(learning_rate=self.Config["Training.LearningRate"], momentum=self.Config["Training.Momentum"])
      else:
        self.Optimizer = optimizers.RMSprop(learning_rate=self.Config["Training.LearningRate"])

    assert self.Optimizer is not None, "Unsupported optimizer"
    print(f"Learning algorithm {self.optimizer_name}")
  # -----------------------------------------------------------------------------------
# =========================================================================================  