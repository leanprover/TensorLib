from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as nps
import numpy as np
import os
import subprocess 
import tempfile
import typing

top = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
tensorlib = top + '/tensorlib' 

def call_lean(file : str) -> str:
    return subprocess.check_output(
       [tensorlib, 'parse-npy', '--write', file],
       stderr=subprocess.STDOUT
       )    

def round_trip(arr : np.ndarray) -> tuple[str, np.ndarray]:
    (_, f) = tempfile.mkstemp(suffix=".npy")
    np.save(f, arr)
    call_lean(f)
    return (f, np.load(f + '.new'))

dim1 = st.integers(min_value=1, max_value=512)
dim2 = st.integers(min_value=1, max_value=10_000)
shape = st.tuples(dim1, dim2) 
@given(
    nps.arrays(dtype=np.float32, shape=shape),
)
def test_decode_inverts_encode(arr):
    print(tensorlib)
    print(arr.shape)
    (f, arr1) = round_trip(arr)
    assert np.array_equal(arr, arr1, equal_nan=True)
    os.remove(f)
    os.remove(f + ".new")