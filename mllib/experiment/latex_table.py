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

class LatexTable(object):
  # --------------------------------------------------------------------------------------------------------------------
  def __init__(self, title, schema):
    self.title = title
    self.schema = schema
    self.schema_dict = dict()
    for sColumn in self.schema:
      self.schema_dict[sColumn] = None
    self.rows = []
    self.report = None
  # --------------------------------------------------------------------------------------------------------------------
  def add_row(self, string_cols, is_column_newline=False):
    sRow = "        "
    for sCol in string_cols:
      sRow += f"{sCol} & "
      if is_column_newline:
        sRow += "\n"
    sRow = sRow.strip()[:-1] + "\\\\"
    self.rows.append(sRow)
  # --------------------------------------------------------------------------------------------------------------------
  def render(self):
    sAlignment = "l" + "c" * (len(self.schema) - 1)

    oRows = []
    oRows.append( "\\setlength{\\tabcolsep}{4pt} % Default value: 6pt")
    oRows.append( "\\begin{table}[!ht]")
    oRows.append( "    \\centering")
    oRows.append(f"    \\caption{{{self.title}}}")
    oRows.append(f"    \\begin{{tabular}}{{{sAlignment}}}")

    # Render the header
    oRows.append( "    \\toprule")
    sHeaderRow = ""
    for sHeaderCol in self.schema:
      sHeaderRow += f"{sHeaderCol} &\n"
    sHeaderRow = sHeaderRow.strip()[:-1] + "\\\\"
    sRow = sHeaderRow.strip()[:-1] + "\\\\"
    oRows.append(sHeaderRow)
    oRows.append("    \\midrule")

    # Render the rows
    for sDataRow in self.rows:
      oRows.append(sDataRow)
    # Render the footer
    oRows.append("    \\bottomrule")

    oRows.append( "    \\end{tabular}")
    oRows.append( "\\end{table}")
    self.report = oRows
    return self
  # --------------------------------------------------------------------------------------------------------------------
  def print(self, is_pausing=False):
    for sRow in self.report:
      print(sRow)
    if is_pausing:
      input()
    return self
  # --------------------------------------------------------------------------------------------------------------------