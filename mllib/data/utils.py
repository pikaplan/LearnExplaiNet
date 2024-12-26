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

def find_max_divisor(number):
    max_divisor = 1

    # Check if the number is divisible by 2
    while number % 2 == 0:
        max_divisor = 2
        number = number // 2

    # Check for other prime factors
    divisor = 3
    while divisor * divisor <= number:
        if number % divisor == 0:
            max_divisor = divisor
            number = number // divisor
        else:
            divisor += 2

    # If the remaining number is greater than 1, it is a prime factor
    if number > 1:
        max_divisor = number

    return max_divisor
  
  
  
def auto_minibatch_size(sample_count, min_size, max_size):
  if sample_count < min_size:
    nBatchSize = sample_count
  else:
    nBatchSize = find_max_divisor(sample_count)
  
  if nBatchSize > max_size:
    nBatchSize = max_size 