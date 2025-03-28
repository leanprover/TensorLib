/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/

import TensorLib.Common
import TensorLib.Iterator

namespace TensorLib

/-!
Shapes and strides in tensors are represented as lists, where the length of the
list is the number of dimensions. For example, a 2 x 3 matrix has a shape of [2, 3].

```
>>> x = np.arange(6).reshape(2, 3)
>>> x
array([[0, 1, 2],
       [3, 4, 5]])
>>> x.shape
(2, 3)
```

(See https://web.mit.edu/dvp/Public/numpybook.pdf for extensive discussion of shape and stride.)

What about the unit element? What is the shape of an empty tensor? For example,
what is the shape of the 1D empty array `[]`? We follow NumPy by defining the shape
as a 1d matrix with 0 elements.

```
>>> np.array([]).shape
(0,)
```

(Assuming we allow 0s in other dimensions, we can shape-check the empty tensor at other shapes, e.g.
`np.array([]).reshape([1,2,3,0,5])` succeeds.)

The only way to have an empty shape in Numpy is as a "scalar array" (or "array scalar" depending on the document.)

    https://numpy.org/doc/stable/reference/arrays.scalars.html

A scalar array is conceptually a boxed scalar with an empty shape

```
>>> np.array(1).shape
()
```

`None` also yields an empty shape, but we ignore this case.

```
>>> np.array(None).shape
()
```

Strides also are empty for scalar arrays.

```
>>> np.array(1).strides
()
```

Shape is independent of the size of the dtype.
TODO: Consider adding Coe instance for List Nat?
-/
structure Shape where
  val : List Nat
deriving BEq, Repr, Inhabited

namespace Shape

instance : ToString Shape where
  toString := reprStr

def empty : Shape := Shape.mk []

def append (shape : Shape) (dims : List Nat) : Shape := Shape.mk (shape.val ++ dims)

--! The number of elements in a tensor. All that's needed is the shape for this calculation.
-- TODO: Put this in the struct?
def count (shape : Shape) : Nat := natProd shape.val

--! Number of dimensions
def ndim (shape : Shape) : Nat := shape.val.length

def map (shape : Shape) (f : List Nat -> List Nat) : Shape := Shape.mk (f shape.val)

def dimIndexInRange (shape : Shape) (dimIndex : DimIndex) :=
  shape.ndim == dimIndex.length &&
  (shape.val.zip dimIndex).all fun (n, i) => i < n

/-!
Strides can be computed from the shape by figuring out how many elements you
need to jump over to get to the next spot and mulitplying by the bytes in each
element.

A given shape can have different strides if the tensor is a view of another
tensor. For example, in a square matrix, the transposed matrix view has the same
shape but the strides change.

Broadcasting does funny things to strides, e.g. the stride can be 0 on a dimension,
so this is just the default case.
-/
def unitStrides (s : Shape) : Strides :=
  let s := s.val
  if H : s.isEmpty then [] else
  let s' := s.reverse
  let rec loop (xs : List Nat) (lastShape lastDimSize : Nat): List Int := match xs with
  | [] => []
  | d :: ds =>
    let rest := loop ds (lastShape * lastDimSize) d
    lastShape * lastDimSize :: rest
  let ok : s' ≠ [] := by
    have H1 : s' = s.reverse := by trivial
    simp [H1]
    simp at H
    exact H
  let res : Strides := 1 :: loop s'.tail 1 (s'.head ok)
  res.reverse

#guard unitStrides (Shape.mk [2]) == [1]
#guard unitStrides (Shape.mk [2, 3]) == [3, 1]
#guard unitStrides (Shape.mk [2, 3, 5, 7]) == [105, 35, 7, 1]

-- Going from position to DimIndex is complicated by the possibility of
-- negative strides.
-- x x x x x x x x x x
--         ^         ^
--         p         s
-- For example, in the 1D array of length 10 above, the start position is at the end.
-- Assume, for example, that we obtained this array from the following sequence
-- # arange(10)[10 : 0 : -1]
-- #
def positionToDimIndex (strides : Strides) (n : Position) : DimIndex :=
  let foldFn acc stride :=
    let (posn, idx) := acc
    let div := (posn / stride).natAbs
    ((posn + (- div * stride)).toNat, div :: idx)
  let (_, idx) := strides.foldl foldFn (n, [])
  idx.reverse

-- TODO: Return `Err Offset` for when the strides and index have different lengths?
def dimIndexToOffset (strides : Strides) (index : DimIndex) : Offset := dot strides (index.map Int.ofNat)

#guard positionToDimIndex [3, 1] 4 == [1, 1]
#guard dimIndexToOffset [3, 1] [1, 1] == 4

-- Just for testing, turning things to lists, etc.
def allDimIndices (shape : Shape) : List DimIndex := Id.run do
  let strides := unitStrides shape
  let count := shape.count
  let mut indices := []
  for i in [0:count] do
    indices := positionToDimIndex strides i :: indices
  return indices.reverse

#guard allDimIndices (Shape.mk [5]) == [[0], [1], [2], [3], [4]]
#guard allDimIndices (Shape.mk [3, 2]) == [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]

-- NumPy supports negative indices, which simply wrap around. E.g. `x[.., -1, ..] = x[.., n-1, ..]` where `n` is the
-- dimension in question. It only supports `-n` to `n`.
def intIndexToDimIndex (shape : Shape) (index : List Int) : Err DimIndex := do
  if shape.ndim != index.length then .error "intsToDimIndex length mismatch" else
  let conv (dim : Nat) (ind : Int) : Err Nat :=
    if 0 <= ind then
      if ind < dim then .ok ind.toNat
      else .error "index out of bounds"
    else if ind < -dim then .error "index out of bounds"
    else .ok (dim + ind).toNat
  (shape.val.zip index).mapM (fun (dim, ind) => conv dim ind)

#guard intIndexToDimIndex (Shape.mk [1, 2, 3]) [0, -1, -1] == (.ok [0, 1, 2])
#guard intIndexToDimIndex (Shape.mk [1, 2, 3]) [0, 1, -2] == (.ok [0, 1, 1])

def belist (shape : Shape) : Iterator.BEList Iterator.NatIter :=
  Iterator.BEList.make $ shape.val.map Iterator.NatIter.make

private def toList (shape : Shape) : List (List Nat) := Iterator.toList shape.belist

#guard (Shape.mk [0, 1]).toList == []
#guard (Shape.mk [1, 0]).toList == []
#guard (Shape.mk [1]).toList == [[0]]
#guard (Shape.mk [3]).toList == [[0], [1], [2]]
#guard (Shape.mk [1, 1]).toList == [[0, 0]]
#guard (Shape.mk [2, 1]).toList == [[0, 0], [1, 0]]
#guard (Shape.mk [1, 1, 1]).toList == [[0, 0, 0]]
#guard (Shape.mk [1, 1, 2]).toList == [[0, 0, 0], [0, 0, 1]]
#guard (Shape.mk [3, 2]).toList == [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]

end Shape
end TensorLib
