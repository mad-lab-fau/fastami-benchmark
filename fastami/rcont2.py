import ctypes
from numpy.ctypeslib import ndpointer
from numpy import ascontiguousarray, int32, empty, array_equal
from numpy.random import randint
from sklearn.metrics.cluster import contingency_matrix
from pathlib import Path

SO_PATH = Path(__file__).resolve().parent / 'rcont2' / 'asa159.so'
ASA159 = ctypes.CDLL(str(SO_PATH))

# void rcont2(int nrow, int ncol, int nrowt[], int ncolt[], int * key,
#             int * seed, int matrix[],  int * ierror)

ASA159.rcont2.argtypes = (ctypes.c_int,
                          ctypes.c_int,
                          ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                          ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                          ctypes.POINTER(ctypes.c_int),
                          ctypes.POINTER(ctypes.c_int),
                          ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                          ctypes.POINTER(ctypes.c_int))


def rcont2(nrowt, ncolt, seed: ctypes.c_int):
    global ASA159

    nrowt = ascontiguousarray(nrowt).astype(int32)
    ncolt = ascontiguousarray(ncolt).astype(int32)

    nrow = ctypes.c_int(len(nrowt))
    ncol = ctypes.c_int(len(ncolt))

    key = ctypes.c_int(0)
    ierror = ctypes.c_int()

    matrix = ascontiguousarray(empty(nrow.value*ncol.value, dtype=int32))

    ASA159.rcont2(
        nrow,
        ncol,
        nrowt,
        ncolt,
        ctypes.byref(key),
        ctypes.byref(seed),
        matrix,
        ctypes.byref(ierror)
    )

    if ierror.value != 0:
        raise RuntimeError(f"Received errorcode {ierror.value} from rcont2.")

    return matrix.reshape((nrow.value, ncol.value), order='F')
