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

def levenshtein_distance(arr1, arr2):
    m = len(arr1)
    n = len(arr2)

    # Create a matrix with dimensions (m+1) x (n+1)
    distance_matrix = np.zeros((m + 1, n + 1), dtype=int)

    # Initialize the first row and column
    distance_matrix[:, 0] = np.arange(m + 1)
    distance_matrix[0, :] = np.arange(n + 1)

    # Fill in the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if arr1[i - 1] == arr2[j - 1]:
                cost = 0
            else:
                cost = 1

            distance_matrix[i, j] = min(
                distance_matrix[i - 1, j] + 1,  # Deletion
                distance_matrix[i, j - 1] + 1,  # Insertion
                distance_matrix[i - 1, j - 1] + cost  # Substitution
            )

    # The bottom-right corner of the matrix represents the Levenshtein distance
    levenshtein_distance = distance_matrix[m, n]
    return levenshtein_distance

# Example usage
arr1 = np.array([14, 2, 3, 4])
arr2 = np.array([1,  2, 3, 5])

distance = levenshtein_distance(arr1, arr2)
print("Levenshtein distance:", distance)
