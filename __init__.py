import os
import subprocess
import sys
import numpy as np
import cv2
from a_cv_imwrite_imread_plus import open_image_in_cv
import numexpr

gooddtypes = [np.uint8, np.uint16, np.uint32, np.uint64]


def _dummyimport():
    import Cython


try:
    from .sort4 import colorfindwithtolerance, unique_boundedpiccount, unique_boundedpic, get_color_coords_parallel, \
        compare_2_pics, get_color_coords
except Exception as e:
    cstring = r"""# distutils: language=c
# distutils: extra_compile_args=/fp:fast /EHsc /Oi /Ot /Oy /Ob3 /GF /Gy /MD /openmp
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: binding=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=True
# cython: overflowcheck.fold=False
# cython: embedsignature=False
# cython: embedsignature.format=c
# cython: cdivision=True
# cython: cdivision_warnings=False
# cython: cpow=True
# cython: c_api_binop_methods=True
# cython: profile=False
# cython: linetrace=False
# cython: infer_types=False
# cython: language_level=3
# cython: c_string_type=bytes
# cython: c_string_encoding=default
# cython: type_version_tag=True
# cython: unraisable_tracebacks=False
# cython: iterable_coroutine=True
# cython: annotation_typing=True
# cython: emit_code_comments=False
# cython: cpp_locals=False
cimport cython
import numpy as np
cimport numpy as np
import cython
from cython.parallel cimport prange
from libc.stdlib cimport abs as c_abs


ctypedef fused urealpic:
    cython.bint
    cython.char
    cython.schar
    cython.uchar
    cython.short
    cython.ushort
    cython.int
    cython.uint
    cython.long
    cython.ulong
    cython.longlong
    cython.ulonglong

    
ctypedef fused ureal:
    cython.uchar
    cython.ushort
    cython.uint
    cython.ulong
    cython.ulonglong    
cpdef void unique_bounded(ureal[:] a,np.npy_bool[:] tmparray,ureal[:] resultarray, cython.uint[:] maxlen ):
    cdef int j = len(a)
    cdef int i
    cdef int lastre =0
    cdef cython.bint bo
    with nogil:
        for i in range(j):
            bo = tmparray[a[i]]
            if not bo:
                tmparray[a[i]]=True
                resultarray[lastre] = a[i]
                lastre+=1
    maxlen[0] = lastre

cpdef void unique_boundedpic(urealpic[:] a,np.npy_bool[:] tmparray,urealpic[:] resultarray, cython.uint[:] maxlen ):
    cdef int j = len(a)
    cdef int i
    cdef int lastre =0
    cdef cython.bint bo
    with nogil:
        for i in range(j):
            bo = tmparray[a[i]]
            if not bo:
                tmparray[a[i]]=True
                resultarray[lastre] = a[i]
                lastre+=1
    maxlen[0] = lastre
cpdef void unique_boundedpiccount(urealpic[:] a,cython.uint[:] tmparray,urealpic[:] resultarray, cython.uint[:] maxlen ):
    cdef Py_ssize_t j = len(a)
    cdef Py_ssize_t i
    cdef Py_ssize_t lastre =0
    cdef cython.bint bo

    with nogil:
        for i in range(j):
            bo = tmparray[a[i]]
            if not bo:

                resultarray[lastre] = a[i]
                lastre+=1
            tmparray[a[i]]+=1
    maxlen[0] = lastre

cpdef void get_color_coords(cython.uchar[:,:,:] arr, cython.uchar[:] r,cython.uchar[:] g,cython.uchar[:] b,cython.int[:] x,cython.int[:] y):
    cdef Py_ssize_t yloop = arr.shape[0]
    cdef Py_ssize_t xloop = arr.shape[1]
    cdef Py_ssize_t ixloop
    cdef Py_ssize_t iyloop
    cdef Py_ssize_t co=0

    with nogil:
        for iyloop in range(xloop):
            for ixloop in range(yloop):
                x[co]=ixloop
                y[co]=iyloop
                r[co]=arr[iyloop][ixloop][2]
                g[co]=arr[iyloop][ixloop][1]
                b[co]=arr[iyloop][ixloop][0]
                co+=1
cpdef void get_color_coords_parallel(cython.uchar[:,:,:] arr, cython.uchar[:] r,cython.uchar[:] g,cython.uchar[:] b,cython.int[:] x,cython.int[:] y):
    cdef int yloop = arr.shape[0]
    cdef int xloop = arr.shape[1]
    cdef int ixloop
    cdef int iyloop
    cdef Py_ssize_t tmpv 

    for iyloop in prange(xloop,nogil=True):
        for ixloop in range(yloop):
            tmpv=iyloop*xloop+ixloop
            x[tmpv]=ixloop
            y[tmpv]=iyloop
            r[tmpv]=arr[iyloop][ixloop][2]
            g[tmpv]=arr[iyloop][ixloop][1]
            b[tmpv]=arr[iyloop][ixloop][0]

cpdef void compare_2_pics(cython.uchar[:,:,:] pic1,cython.uchar[:,:,:] pic2,cython.uchar rmax, cython.uchar
 gmax,cython.uchar bmax, cython.uchar[:] r0,cython.uchar[:] g0,cython.uchar[:] b0, cython.uchar[:] r,cython.uchar[:] g,
 cython.uchar[:] b, cython.int[:] dr,cython.int[:] dg,cython.int[:] db,cython.int[:] x,cython.int[:] y,np.npy_bool[:] indi):
    cdef int yloop = pic1.shape[0]
    cdef int xloop = pic1.shape[1]
    cdef int ixloop
    cdef int iyloop
    cdef Py_ssize_t tmpv 
    cdef int checkval 

    cdef bint vale1 
    cdef bint vale2 
    cdef bint vale3 
    cdef int absvale1 
    cdef int absvale2 
    cdef int absvale3 
    for iyloop in prange(xloop,nogil=True):
        for ixloop in range(yloop):
            tmpv=iyloop*xloop+ixloop
            checkval = pic1[iyloop][ixloop][2] - pic2[iyloop][ixloop][2]
            absvale1= c_abs(checkval)
            vale1 = absvale1 <= rmax
            if not vale1:
                continue
            checkval = pic1[iyloop][ixloop][1] - pic2[iyloop][ixloop][1]
            absvale2= c_abs(checkval)
            vale2 =absvale2 <= gmax
            if not vale2:
                continue
            checkval = pic1[iyloop][ixloop][0] - pic2[iyloop][ixloop][0]
            absvale3= c_abs(checkval)

            vale3 = absvale3 <= bmax
            if not vale3:
                continue
            x[tmpv]=ixloop
            y[tmpv]=iyloop
            r[tmpv]=pic1[iyloop][ixloop][2]
            g[tmpv]=pic1[iyloop][ixloop][1]
            b[tmpv]=pic1[iyloop][ixloop][0]
            r0[tmpv]=pic2[iyloop][ixloop][2]
            g0[tmpv]=pic2[iyloop][ixloop][1]
            b0[tmpv]=pic2[iyloop][ixloop][0]
            dr[tmpv]=absvale1
            dg[tmpv]=absvale2
            db[tmpv]=absvale3
            indi[tmpv]=True

cpdef void colorfindwithtolerance(cython.uchar[:,:,:] pic1,cython.uchar[:,:] pic2,cython.uchar rmax, cython.uchar
 gmax,cython.uchar bmax, cython.uchar[:] r0,cython.uchar[:] g0,cython.uchar[:] b0, cython.uchar[:] r,cython.uchar[:] g,
 cython.uchar[:] b, cython.int[:] dr,cython.int[:] dg,cython.int[:] db,cython.int[:] y,np.npy_bool[:] indi):
    cdef int yloop = pic1.shape[0]
    cdef int xloop = pic1.shape[1]
    cdef int ixloop
    cdef int iyloop
    cdef Py_ssize_t tmpv 
    cdef int checkval 

    cdef bint vale1 
    cdef bint vale2 
    cdef bint vale3 
    cdef int absvale1 
    cdef int absvale2 
    cdef int absvale3 
    cdef int loopcors = pic2.shape[0]
    cdef int cor
    cdef bint alreadydone 
    for iyloop in prange(xloop,nogil=True):
        for ixloop in range(yloop):
                tmpv=iyloop*(xloop)+ixloop
                alreadydone = indi[tmpv]
                if alreadydone:
                    continue
                for cor in range(loopcors):

                    checkval = pic1[iyloop][ixloop][2] - pic2[cor][2]
                    absvale1= c_abs(checkval)
                    vale1 = absvale1 <= rmax
                    if not vale1:
                        continue
                    checkval = pic1[iyloop][ixloop][1] - pic2[cor][1]
                    absvale2= c_abs(checkval)
                    vale2 =absvale2 <= gmax
                    if not vale2:
                        continue
                    checkval = pic1[iyloop][ixloop][0] - pic2[cor][0]
                    absvale3= c_abs(checkval)

                    vale3 = absvale3 <= bmax
                    if not vale3:
                        continue
                    y[tmpv]=tmpv
                    r[tmpv]=pic1[iyloop][ixloop][2]
                    g[tmpv]=pic1[iyloop][ixloop][1]
                    b[tmpv]=pic1[iyloop][ixloop][0]
                    r0[tmpv]=pic2[cor][2]
                    g0[tmpv]=pic2[cor][1]
                    b0[tmpv]=pic2[cor][0]
                    dr[tmpv]=absvale1
                    dg[tmpv]=absvale2
                    db[tmpv]=absvale3
                    indi[tmpv]=True
                    break


"""
    pyxfile = f"sort4.pyx"
    pyxfilesetup = f"sort4compiled_setup.py"

    dirname = os.path.abspath(os.path.dirname(__file__))
    pyxfile_complete_path = os.path.join(dirname, pyxfile)
    pyxfile_setup_complete_path = os.path.join(dirname, pyxfilesetup)

    if os.path.exists(pyxfile_complete_path):
        os.remove(pyxfile_complete_path)
    if os.path.exists(pyxfile_setup_complete_path):
        os.remove(pyxfile_setup_complete_path)
    with open(pyxfile_complete_path, mode="w", encoding="utf-8") as f:
        f.write(cstring)
    numpyincludefolder = np.get_include()
    compilefile = (
            """
	from setuptools import Extension, setup
	from Cython.Build import cythonize
	ext_modules = Extension(**{'py_limited_api': False, 'name': 'sort4', 'sources': ['sort4.pyx'], 'include_dirs': [\'"""
            + numpyincludefolder
            + """\'], 'define_macros': [], 'undef_macros': [], 'library_dirs': [], 'libraries': [], 'runtime_library_dirs': [], 'extra_objects': [], 'extra_compile_args': [], 'extra_link_args': [], 'export_symbols': [], 'swig_opts': [], 'depends': [], 'language': None, 'optional': None})

	setup(
		name='sort4',
		ext_modules=cythonize(ext_modules),
	)
			"""
    )
    with open(pyxfile_setup_complete_path, mode="w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [x.lstrip().replace(os.sep, "/") for x in compilefile.splitlines()]
            )
        )
    subprocess.run(
        [sys.executable, pyxfile_setup_complete_path, "build_ext", "--inplace"],
        cwd=dirname,
        shell=True,
        env=os.environ.copy(),
    )
    try:
        from .sort4 import colorfindwithtolerance, unique_boundedpiccount, unique_boundedpic, get_color_coords_parallel, \
            compare_2_pics, get_color_coords
    except Exception as fe:
        sys.stderr.write(f'{fe}')
        sys.stderr.flush()


def get_unique_colors(pic):
    r"""
    Get unique colors from an image.

    Parameters:
    - pic (numpy array): Input image.

    Returns:
    numpy array: Array of unique colors.
    """
    x = open_image_in_cv(pic, channels_in_output=3)

    arr = x.reshape(-1, x.shape[-1])
    a = numexpr.evaluate('(c1 << 16) + (c2 << 8) + c3', global_dict={},
                         local_dict={'c1': arr[..., 0], 'c2': arr[..., 1], 'c3': arr[..., 2]})
    tmparray = np.zeros(256 * 256 * 256, dtype=bool)
    resultarray = np.zeros_like(a)
    maxlen = np.zeros(1, dtype=np.uint32)
    unique_boundedpic(a, tmparray, resultarray, maxlen)
    bx = resultarray[:maxlen[0]]
    return np.vstack([bx & 255, bx >> 8 & 255, bx >> 16 & 255]).T


def _corcount(x):
    r"""
    Internal function to count occurrences of colors in an image.

    Parameters:
    - x (numpy array): Input image.

    Returns:
    tuple: Arrays of unique colors, their occurrences, and a temporary array for counting.
    """

    arr = x.reshape(-1, x.shape[-1])
    a = numexpr.evaluate('(c1 << 16) + (c2 << 8) + c3', global_dict={},
                         local_dict={'c1': arr[..., 0], 'c2': arr[..., 1], 'c3': arr[..., 2]})
    tmparray = np.zeros(256 * 256 * 256, dtype=np.uint32)
    resultarray = np.zeros_like(a)
    maxlen = np.zeros(1, dtype=np.uint32)
    unique_boundedpiccount(a, tmparray, resultarray, maxlen)
    return resultarray, maxlen, tmparray


def count_colors(pic):
    r"""
    Count the occurrences of each color in an image.

    Parameters:
    - pic (numpy array): Input image.

    Returns:
    numpy array: Array of unique colors and their occurrences.
    """
    x = open_image_in_cv(pic, channels_in_output=3)

    resultarray, maxlen, tmparray = _corcount(x)
    bx = resultarray[:maxlen[0]]
    return np.vstack([bx & 255, bx >> 8 & 255, bx >> 16 & 255, tmparray[bx]]).T


def get_most_frequent_colors(pic):
    """
    Get the most frequent colors in an image.

    Parameters:
    - pic (numpy array): Input image.

    Returns:
    numpy array: Array of most frequent colors.
    """
    x = open_image_in_cv(pic, channels_in_output=3)
    resultarray, maxlen, tmparray = _corcount(x)
    bx = resultarray[:maxlen[0]]
    co = tmparray[bx]
    maxcount = np.max(co)
    cx = bx[np.where(co == maxcount)]
    fia = np.zeros_like(cx)
    fia[:] = maxcount
    return np.vstack([cx & 255, cx >> 8 & 255, cx >> 16 & 255, fia]).T


def get_rgb_coords(pic):
    """
    Get RGB coordinates of each pixel in an image.

    Parameters:
    - pic (numpy array): Input image.

    Returns:
    numpy array: Arrays of red, green, blue channel values, x-coordinates, and y-coordinates.
    """
    x = open_image_in_cv(pic, channels_in_output=3)
    size = x.shape[0] * x.shape[1]
    r = np.zeros(size, dtype=np.uint8)
    g = np.zeros(size, dtype=np.uint8)
    b = np.zeros(size, dtype=np.uint8)
    xc = np.zeros(size, dtype=np.int32)
    yc = np.zeros(size, dtype=np.int32)
    get_color_coords(x, r, g, b, xc, yc)
    return np.stack([r, g, b, yc, xc], axis=1)


def get_rgb_coords_parallel(pic):
    """
    Get RGB coordinates of each pixel in an image in parallel.

    Parameters:
    - pic (numpy array): Input image.

    Returns:
    numpy array
    """
    x = open_image_in_cv(pic, channels_in_output=3)

    size = x.shape[0] * x.shape[1]
    r = np.zeros(size, dtype=np.uint8)
    g = np.zeros(size, dtype=np.uint8)
    b = np.zeros(size, dtype=np.uint8)
    xc = np.zeros(size, dtype=np.int32)
    yc = np.zeros(size, dtype=np.int32)
    get_color_coords_parallel(x, r, g, b, xc, yc)
    return np.stack([r, g, b, yc, xc], axis=1)


def compare_rgb_values_of_2_pics(

        img1,
        img2,
        rmax=0,
        gmax=0,
        bmax=0,
):
    r"""
    Compare RGB values of two images within specified tolerances.

    Parameters:
    - pic1 (numpy array): First input image.
    - pic2 (numpy array): Second input image.
    - rmax (int): Maximum tolerance for the red channel.
    - gmax (int): Maximum tolerance for the green channel.
    - bmax (int): Maximum tolerance for the blue channel.

    Returns:
    numpy array
    """
    pic1 = open_image_in_cv(img1, channels_in_output=3)
    pic2 = open_image_in_cv(img2, channels_in_output=3)

    if pic1.shape != pic2.shape:
        if np.product(pic1.shape) > np.product(pic2.shape):
            pic1 = cv2.resize(pic1, pic2.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        elif np.product(pic1.shape) < np.product(pic2.shape):
            pic2 = cv2.resize(pic2, pic1.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

    size = pic1.shape[0] * pic1.shape[1]

    r0 = np.zeros(size, dtype=np.uint8)
    g0 = np.zeros(size, dtype=np.uint8)
    b0 = np.zeros(size, dtype=np.uint8)
    r = np.zeros(size, dtype=np.uint8)
    g = np.zeros(size, dtype=np.uint8)
    b = np.zeros(size, dtype=np.uint8)
    dr = np.zeros(size, dtype=np.int32)
    dg = np.zeros(size, dtype=np.int32)
    db = np.zeros(size, dtype=np.int32)
    xc = np.zeros(size, dtype=np.int32)
    yc = np.zeros(size, dtype=np.int32)
    indi = np.zeros(size, dtype=bool)
    compare_2_pics(
        pic1,
        pic2,
        rmax,
        gmax,
        bmax,
        r0,
        g0,
        b0,
        r,
        g,
        b,
        dr,
        dg,
        db,
        xc,
        yc,
        indi
    )

    valida = np.nonzero(indi)[0]

    return np.stack([

        r0[valida],
        g0[valida],
        b0[valida],
        r[valida],
        g[valida],
        b[valida],
        dr[valida],
        dg[valida],
        db[valida],
        xc[valida],
        yc[valida], ], axis=1)


def find_color_ranges(im, colors, rmax=0,
                      gmax=0,
                      bmax=0):
    r"""
    Find color ranges in an image based on given colors and tolerance.

    Parameters:
    - pic (numpy array): Input image.
    - colors (numpy array): Array of target colors.
    - rmax (int): Maximum tolerance for the red channel.
    - gmax (int): Maximum tolerance for the green channel.
    - bmax (int): Maximum tolerance for the blue channel.

    Returns:
    numpy array
    """
    pic = open_image_in_cv(im, channels_in_output=3)
    if not isinstance(colors, np.ndarray):
        colors = np.array(colors)
    if colors.dtype != np.uint8:
        colors = colors.astype(np.uint8)

    size = pic.shape[0] * pic.shape[1]

    r0 = np.zeros(size, dtype=np.uint8)
    g0 = np.zeros(size, dtype=np.uint8)
    b0 = np.zeros(size, dtype=np.uint8)
    r = np.zeros(size, dtype=np.uint8)
    g = np.zeros(size, dtype=np.uint8)
    b = np.zeros(size, dtype=np.uint8)
    dr = np.zeros(size, dtype=np.int32)
    dg = np.zeros(size, dtype=np.int32)
    db = np.zeros(size, dtype=np.int32)
    yc = np.zeros(size, dtype=np.int32)
    indi = np.zeros(size, dtype=bool)
    colorfindwithtolerance(
        pic,
        colors,
        rmax,
        gmax,
        bmax,
        r0,
        g0,
        b0,
        r,
        g,
        b,
        dr,
        dg,
        db,
        yc,
        indi
    )

    valida = np.nonzero(indi)[0]
    o = ([

        r0[valida],
        g0[valida],
        b0[valida],
        r[valida],
        g[valida],
        b[valida],
        dr[valida],
        dg[valida],
        db[valida],
        yc[valida], ])
    return np.stack([*o[:-1], *np.divmod(o[-1], pic.shape[1])], axis=1)
