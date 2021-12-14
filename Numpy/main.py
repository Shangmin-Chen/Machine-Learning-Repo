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

# get a specific column
print(a[:,2])

# [start_index:end_index:step_size]
print(a[0, 1:6:2])

# you can change these index aswell

# Initializing different types of arrays

# all 0s matrix
print()
print(np.zeros((2,3)))

# all 1s matrix
print()
print(np.ones((4,2,2), dtype="int32"))

# any other number
print()
print(np.full((2,2),99))

# full like
print()
print(np.full(a.shape, 4))

# random deci
print(np.random.rand(4,4,3))
print()

#random like
print()
print(np.random.random_sample(a.shape))

# random int (start,end,size)
print()
print(np.random.randint(1, 7, size=(3,3)))

# identity matrix
print()
print(np.identity(3))

# repeating
print()
arr = np.array([[1,2,3]])
r1 = np.repeat(arr,3,axis=0)
print(r1)

"""MATH"""
print()
print(a)
print()
print(a+2)
print()
print(np.cos(a))

# linear algebra
print("\n\n\nmulti matrix")

a = np.ones((2,3))
print(a)

b = np.full((3,2),2)
print(b)

print(np.matmul(a,b))

#statistics
print("\n\n\nstatistics")

stats = np.array([[1,2,3],[4,5,6]])

print(np.min(stats))
print(np.max(stats, axis=0))
print(np.max(stats, axis=1))
print(np.max(stats))
print(np.sum(stats))

#reorganize
"""
np.array().reshape(())
np.vstack([v1,v2])
np.hstack([h1, h2])
"""

#load data
"""

filedata = np.genfromtxt("data.txt"), delimiter=",")
filedata = filedata.astype("int32")

print(filedata)

"""
