# Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Released under Apache 2.0 license as described in the file LICENSE.
# Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as nps
import numpy as np
import os
import subprocess
import tempfile
import typing

top = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
tensorlib = top + '/bin/tensorlib'

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
@settings(deadline=None) # The very first call is flakey wrt timing, e.g. 2s vs 100ms. 
def test_numpy_save_load(arr):
    print(tensorlib)
    print(arr.shape)
    (f, arr1) = round_trip(arr)
    assert np.array_equal(arr, arr1, equal_nan=True)
    os.remove(f)
    os.remove(f + ".new")
