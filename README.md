# TensorLib

TensorLib is a numerical computing library in Lean4 that provides NumPy-like multidimensional tensors, numerical data types, and tensor operations. A primary goal of TensorLib is to provide a foundation for verified numerical computing.

In addition to providing executable tensor computations, TensorLib is icludes support to formally reasoning about numerical programs. By representing tensor operations directly in Lean, TensorLib makes it possible to state and prove properties about data types, numerical operations, and mixed-precision computations within the same system in which the computations are defined.

## Features

* multidimensional tensors with shapes and strides
* tensor indexing and slicing
* reshaping and transposing tensors
* NumPy-style broadcasting
* numerical operations over tensors
* integer and floating-point data types
* mixed-precision computations
* FP8 data types
* NumPy `.npy` file parsing and serialization
* testing tensor behavior against NumPy
* formal reasoning about tensor and numerical computations in Lean

## Verification
Tensor operations are implemented directly in Lean, allowing properties of these operations to be expressed and proved using Lean's prover. This makes it possible to reason about properties such as:

* tensor shapes and indexing
* broadcasting behavior
* data type conversions
* numerical representations
* floating-point and mixed-precision computations
* correctness properties of tensor operations

## Supported data types

TensorLib supports a range of integer and floating-point formats.

### Integer types

* `bool`
* `int8`
* `int16`
* `int32`
* `int64`
* `uint8`
* `uint16`
* `uint32`
* `uint64`
* `int64`

### Floating-point types

* `float8_e2m5`
* `float8_e3m4`
* `float8_e4m3`
* `float8_e5m2`
* `float8_e8m0`
* float16`
* `bfloat16`
* `float32`
* `float64`

## Installation

If Lean is installed using `elan`, cloning the repository will automatically select the Lean version specified by the project.

```bash
git clone https://github.com/leanprover/TensorLib.git
cd TensorLib
lake build
```

## Basic usage

Import TensorLib:

```lean
import TensorLib

open TensorLib
```

A tensor can be created using `Tensor.arange!` and reshaped into a multidimensional array.

```lean
def x :=
  (Tensor.arange! Dtype.uint8 6).reshape! (Shape.mk [2, 3])

#eval x.toNatTree!.format!
```

This creates a tensor containing the values

```text
0 1 2
3 4 5
```

with shape `(2, 3)`.

### Broadcasting

TensorLib supports NumPy-style broadcasting.

```lean
def x :=
  (Tensor.arange! Dtype.uint8 6).reshape! (Shape.mk [2, 3])

def y :=
  x.broadcastTo! (Shape.mk [2, 2, 3])
```

Here, `x` has shape `(2, 3)` and is broadcast to shape `(2, 2, 3)`.

## NumPy interoperability

TensorLib includes support for NumPy's `.npy` file format.

A NumPy array can be parsed from the command line using:

```bash
lake exe tensorlib parse-npy path/to/file.npy
```

The parsed tensor can also be written back to disk:

```bash
lake exe tensorlib parse-npy --write path/to/file.npy
```

NumPy interoperability is also useful for testing TensorLib's executable behavior against an established numerical computing implementation.

## Testing

TensorLib contains both Lean tests and Python tests.

To run the tests install Python dependencies using:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run the test suite with:

```bash
lake exe tensorlib test
```

The tests include checks of tensor operations and comparisons with Python and NumPy implementations.

## Project structure

The main TensorLib implementation is located in `TensorLib/`.

Some of the main modules include:

* `Tensor.lean` — tensor representation and related operations
* `Dtype.lean` — numerical data types, conversions, and arithmetic
* `Float.lean` — floating-point representations and operations
* `Shape.lean` — tensor shapes
* `Broadcast.lean` — tensor broadcasting
* `Index.lean` — tensor indexing
* `Slice.lean` — tensor slicing
* `Ufunc.lean` — element-wise tensor operations
* `Npy.lean` — NumPy `.npy` file support
* `LOrd.lean` - Typeclass for floating points defined as a partial order
* `MixedPrec.lean` — mixed-precision operations

## Development

After making changes to the library, build the project with:

```bash
lake build
```

and run the test suite with:

```bash
lake exe tensorlib test
```

When adding new tensor operations or data types, tests should be added where appropriate. Properties of new operations can additionally be stated and proved in Lean when formal guarantees are required.

## License

TensorLib is licensed under the Apache License 2.0. See [`LICENSE`](LICENSE) for details.
