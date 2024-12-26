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
from scipy.stats import spearmanr

def rank_similarity(true_set, predicted_set):
  nItemCount = len(true_set)
  assert len(predicted_set) == nItemCount, f"Invalid set size. Should be {nItemCount}"
  discount = 1 / (np.log2(np.arange(nItemCount) + 2))
  
  nEqual = np.array(true_set[:] == predicted_set[:]).astype(np.float32)
  nNorm = np.dot(discount, np.ones((nItemCount), np.float32))
  nVal  = np.dot(discount, nEqual)
  
  return nVal / nNorm
  

def ndcg(rel_true, rel_pred, form="linear"):
    """ Returns normalized Discounted Cumulative Gain
    Args:
        rel_true (1-D Array): relevance lists for particular user, (n_songs,)
        rel_pred (1-D Array): predicted relevance lists, (n_pred,)
        form (string): two types of nDCG formula, 'linear' or 'exponential'
    Returns:
        ndcg (float): normalized discounted cumulative gain score [0, 1]
    """
    rel_true = np.sort(rel_true)[::-1]
    p = min(len(rel_true), len(rel_pred))
    discount = 1 / (np.log2(np.arange(p) + 2))

    if form == "linear":
        idcg = np.sum(rel_true[:p] * discount)
        dcg = np.sum(rel_pred[:p] * discount)
    elif form == "exponential" or form == "exp":
        idcg = np.sum([2**x - 1 for x in rel_true[:p]] * discount)
        dcg = np.sum([2**x - 1 for x in rel_pred[:p]] * discount)
    else:
        raise ValueError("Only supported for two formula, 'linear' or 'exp'")

    return dcg / idcg
  
def ndcg_score(y_true, y_pred, k=None):
    # Sort the predicted scores and corresponding true labels in descending order
    sorted_indices = np.argsort(y_pred)[::-1]
    sorted_labels = y_true[sorted_indices]

    # Calculate the DCG
    dcg = np.sum(sorted_labels / np.log2(np.arange(2, sorted_labels.size + 2)))

    # Calculate the IDCG (Ideal DCG)
    ideal_sorted_labels = np.sort(y_true)[::-1]
    if k is not None:
        ideal_sorted_labels = ideal_sorted_labels[:k]
    idcg = np.sum(ideal_sorted_labels / np.log2(np.arange(2, ideal_sorted_labels.size + 2)))

    # Calculate NDCG
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return ndcg
    
# Copyright 2016 Krysta M Bouzek
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


"""
Implementation of normalized discounted cumulative gain.

Handy for testing ranking algorithms.

https://en.wikipedia.org/wiki/Discounted_cumulative_gain
"""

def cum_gain(relevance):
    """
    Calculate cumulative gain.
    This ignores the position of a result, but may still be generally useful.

    @param relevance: Graded relevances of the results.
    @type relevance: C{seq} or C{numpy.array}
    """

    if relevance is None or len(relevance) < 1:
        return 0.0

    return np.asarray(relevance).sum()


def dcg(relevance, alternate=True):
    """
    Calculate discounted cumulative gain.

    @param relevance: Graded and ordered relevances of the results.
    @type relevance: C{seq} or C{numpy.array}
    @param alternate: True to use the alternate scoring (intended to
    place more emphasis on relevant results).
    @type alternate: C{bool}
    """

    if relevance is None or len(relevance) < 1:
        return 0.0

    rel = np.asarray(relevance)
    p = len(rel)

    if alternate:
        # from wikipedia: "An alternative formulation of
        # DCG[5] places stronger emphasis on retrieving relevant documents"

        log2i = np.log2(np.asarray(range(1, p + 1)) + 1)
        return ((np.power(2, rel) - 1) / log2i).sum()
    else:
        log2i = np.log2(range(2, p + 1))
        return rel[0] + (rel[1:] / log2i).sum()


def idcg(relevance, alternate=True):
    """
    Calculate ideal discounted cumulative gain (maximum possible DCG).

    @param relevance: Graded and ordered relevances of the results.
    @type relevance: C{seq} or C{numpy.array}
    @param alternate: True to use the alternate scoring (intended to
    place more emphasis on relevant results).
    @type alternate: C{bool}
    """

    if relevance is None or len(relevance) < 1:
        return 0.0

    # guard copy before sort
    rel = np.asarray(relevance).copy()
    rel.sort()
    return dcg(rel[::-1], alternate)


def ndcg_rel(relevance, nranks, alternate=True):
    """
    Calculate normalized discounted cumulative gain.

    @param relevance: Graded and ordered relevances of the results.
    @type relevance: C{seq} or C{numpy.array}
    @param nranks: Number of ranks to use when calculating NDCG.
    Will be used to rightpad with zeros if len(relevance) is less
    than nranks
    @type nranks: C{int}
    @param alternate: True to use the alternate scoring (intended to
    place more emphasis on relevant results).
    @type alternate: C{bool}
    """
    if relevance is None or len(relevance) < 1:
        return 0.0

    if (nranks < 1):
        raise Exception('nranks < 1')

    rel = np.asarray(relevance)
    pad = max(0, nranks - len(rel))

    # pad could be zero in which case this will no-op
    rel = np.pad(rel, (0, pad), 'constant')

    # now slice downto nranks
    rel = rel[0:min(nranks, len(rel))]

    ideal_dcg = idcg(rel, alternate)
    if ideal_dcg == 0:
        return 0.0

    return dcg(rel, alternate) / ideal_dcg

    
if __name__ == "__main__":
  oList  = np.array([8, 1, 4, 0])
  oList0 = np.array([8, 0, 1, 5])
  oList1 = np.array([8, 1, 4, 5])
  oList2 = np.array([8, 1, 2, 0])
  oList3 = np.array([1, 8, 4, 0])
  oList4 = np.array([1, 8, 4, 5])

  print(rank_similarity(oList, oList) , spearmanr(oList, oList))
  print(rank_similarity(oList, oList0) , spearmanr(oList, oList0))  
  print(rank_similarity(oList, oList1), spearmanr(oList, oList1))
  print(rank_similarity(oList, oList2), spearmanr(oList, oList2))
  print(rank_similarity(oList, oList3), spearmanr(oList, oList3))
  print(rank_similarity(oList, oList4), spearmanr(oList, oList4))
      
  
  
  
  
  
  
  