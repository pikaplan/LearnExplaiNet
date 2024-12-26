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

import numpy as np
import xlwt

class MetricsTable(object):
  def __init__(self, model_group, metric_names):
    self.model_group = model_group
    self.metrics = dict()
    self.group_name = None
    for nMetricIndex,sMetric in enumerate(metric_names):
      oNewList = []
      self.metrics[sMetric] = oNewList

  @property
  def row_count(self):
    return len(self.metrics[list(self.metrics.keys())[0]])

  def metric_values(self, metric_name):
    nSerie = np.asarray(self.metrics[metric_name], np.float32)
    return nSerie

  def metric_min_max(self, metric_name):
    nSerie = self.metric_values(metric_name)
    nMin = np.min(nSerie)
    nMax = np.max(nSerie)
    return nMin, nMax

  def metric_values_normalized(self, metric_name):
    nSerie = np.asarray(self.metrics[metric_name], np.float32)
    nMin = np.min(nSerie)
    nMax = np.max(nSerie)
    return (nSerie - nMin)/(nMax - nMin)

  def add_experiment(self, metric_name, metric_value, is_auto_schema=False):
    if is_auto_schema:
      if metric_name not in self.metrics:
        oNewList = []
        self.metrics[metric_name] = oNewList

    if metric_name in self.metrics:
      if isinstance(metric_value, list):
        metric_value = np.array(metric_value, np.float32)
      oSerie = self.metrics[metric_name]
      oSerie.append(metric_value)

  def last_experiment_metrics_row(self):
    dMetricsRow = dict()
    for sMetricName, nMetricSeries in self.metrics.items():
      if len(nMetricSeries) > 0:
        dMetricsRow[sMetricName] = nMetricSeries[-1]
      else:
        dMetricsRow[sMetricName] = []
    return dMetricsRow

  def export_to_xlsx(self, filename, is_overwriting=True):
    oExcelWorkbook = xlwt.Workbook()
    oExcelSheet = oExcelWorkbook.add_sheet(self.model_group.upper())

    oHeaders = list(self.metrics.keys())
    for nColIndex, sMetricKey in enumerate(oHeaders):
      oExcelSheet.write(0, nColIndex, sMetricKey)

    nRowStartIndex = 1
    for nColIndex, sMetricKey in enumerate(oHeaders):
      oSerie = self.metrics[sMetricKey]
      if oSerie is not None:
        for nRowIndex, nMetricValue in enumerate(oSerie):
          if nMetricValue is not None:
            if isinstance(nMetricValue, np.ndarray):
              oExcelSheet.write(nRowStartIndex + nRowIndex , nColIndex, str(nMetricValue))
            else:
              oExcelSheet.write(nRowStartIndex + nRowIndex , nColIndex, nMetricValue)

    if os.path.exists(filename):
      if is_overwriting:
        os.remove(filename)
        oExcelWorkbook.save(filename)
      else:
        print(r"Not overwriting workbook file {filename")
    else:
      oExcelWorkbook.save(filename)


