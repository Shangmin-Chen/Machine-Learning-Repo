"""Lists vs NumPy"""

"""Numpy is faster because it uses Fixed Type"""

# NumPy binary turns into int 32, int 16, int 8,
# Lists binary turns into size, reference count, object type, object value
# each of those is its individual byte section, a lot more bytes

"""NumPy is faster to read less bytes of memory"""
# No type checking when iterating through objects

"""Numpy is faster because of Contiguous Memory"""

# SIMD Vector Processing. perform computing on memory together
# Caches faster.

"""Application of Numpy"""
# Mathematics (MATLAB Replacement)
# Plotting (Matplotlib)
# Backend (Pandas, Connect 4, Digital Photography)
# Machine Learning

"""CODING Basics"""

import numpy as np

a = np.array([1,2,3])
print(a)

b = np.array([[9.0,8.0,7.0],[6.0,5.0,4.0]])
print(b)
# setup an array with a type, you specify data type to be as low as it can to make your code efficient
c = np.array([1,2,3], dtype="int16")
print(a)

# dimensions
print(b.ndim)
# shape
print(b.shape)
# type
print(b.dtype)
# byte size - this is in proportion to to type, since its float 64 then its 8 bytes,
# if its int 32 just like var a then its 4 bytes
print(b.itemsize)
# total size 6*8
print(b.size * b.itemsize)
print(b.nbytes)

"""Accessing/ Changing specific elements, rows, columns, etc"""
a = np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
print(a)
print(a[1,5])
# get a specific row
print(a[0,:])




