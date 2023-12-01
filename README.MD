# Provides functions (Cython - C) for color analysis in images, including finding unique colors, counting occurrences, and comparing images

## pip install cythonimagetools

### Tested against Windows / Python 3.11 / Anaconda

## Cython (and a C/C++ compiler) must be installed


```python
import numpy as np
from cythonimagetools import get_unique_colors, count_colors, get_most_frequent_colors, get_rgb_coords, \
    get_rgb_coords_parallel, compare_rgb_values_of_2_pics, find_color_ranges

# Generate a random image
np.random.seed(0)
x = np.random.randint(0, 9, (1000, 1000, 3))

# Get unique colors in the image
uniquecolors = get_unique_colors(x)
print(uniquecolors)

# Count the occurrence of each color in the image
allcolors = count_colors(x)
print(allcolors)

# Get the most frequent colors in the image
most_freq_colors = get_most_frequent_colors(x)
print(most_freq_colors)

# Get RGB coordinates of each pixel in the image
rgb_coords = get_rgb_coords(x)
print(rgb_coords)

# Get RGB coordinates in parallel
rgb_coords_parallel = get_rgb_coords_parallel(x)
print(rgb_coords_parallel)

# Compare RGB values of two images within a specified tolerance
pic1 = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
pic2 = np.random.randint(0, 200, (1000, 1000, 3), dtype=np.uint8)
ru = compare_rgb_values_of_2_pics(pic1, pic2, rmax=10, gmax=10, bmax=10)
print(ru)

# Find color ranges in an image based on given colors and tolerance
colors = np.array([[2, 7, 4], [4, 7, 2]], dtype=np.uint8)
foundcolors = find_color_ranges(x, colors, rmax=1, gmax=1, bmax=2)
print(foundcolors)


print(uniquecolors)
print('--------------------')

print(allcolors)
print('--------------------')

print(most_freq_colors)
print('--------------------')

print(rgb_coords)
print('--------------------')

print(rgb_coords_parallel)
print('--------------------')

print(ru)
print('--------------------')

print(colors)
print('--------------------')

print(foundcolors)
print('--------------------')


# [[3 0 5]
#  [3 7 3]
#  [4 2 5]
#  ...
#  [2 0 7]
#  [3 8 5]
#  [2 4 1]]
# --------------------
# [[   3    0    5 1402]
#  [   3    7    3 1380]
#  [   4    2    5 1369]
#  ...
#  [   2    0    7 1413]
#  [   3    8    5 1343]
#  [   2    4    1 1405]]
# --------------------
# [[   2    7    4 1473]]
# --------------------
# [[  3   0   5   0   0]
#  [  3   7   3   0   1]
#  [  4   2   5   0   2]
#  ...
#  [  6   1   1 999 997]
#  [  0   6   4 999 998]
#  [  8   0   6 999 999]]
# --------------------
# [[  3   0   5   0   0]
#  [  3   7   3   0   1]
#  [  4   2   5   0   2]
#  ...
#  [  6   1   1 999 997]
#  [  0   6   4 999 998]
#  [  8   0   6 999 999]]
# --------------------
# [[143  76  62 ...   5 920   0]
#  [153  80  38 ...   9 190   3]
#  [ 42 198  93 ...   8 332   3]
#  ...
#  [ 71  12  64 ...   5   0 995]
#  [174  85 160 ...   7 107 996]
#  [193  37 160 ...   0 817 997]]
# --------------------
# [[2 7 4]
#  [4 7 2]]
# --------------------
# [[  4   7   2 ...   1   0   1]
#  [  4   7   2 ...   2   0  22]
#  [  4   7   2 ...   2   0  24]
#  ...
#  [  4   7   2 ...   2 999 956]
#  [  4   7   2 ...   2 999 970]
#  [  2   7   4 ...   0 999 987]]
# --------------------
```