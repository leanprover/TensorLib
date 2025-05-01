/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/

import TensorLib.Broadcast
import TensorLib.Common
import TensorLib.Dtype
import TensorLib.Shape
import TensorLib.Slice
import TensorLib.Tensor

open TensorLib.Slice (Iter)

/-
There are several types of indexing in NumPy.

    https://numpy.org/doc/stable/user/basics.indexing.html

We handle basic indexing and some types of advanced indexing.

Theorems to prove (taken from NumPy docs):

1. Basic slicing with more than one non-: entry in the slicing tuple,
   acts like repeated application of slicing using a single non-: entry,
   where the non-: entries are successively taken (with all other non-: entries replaced by :).
   Thus, x[ind1, ..., ind2,:] acts like x[ind1][..., ind2, :] under basic slicing.

2. Advanced indices always are broadcast and iterated as one:
   result[i_1, ..., i_M] == x[ind_1[i_1, ..., i_M], ind_2[i_1, ..., i_M],
                           ..., ind_N[i_1, ..., i_M]]
3. ...TODO...
-/

namespace TensorLib
namespace Index

-- Parsed form. We will simplify some of the redundancy here to yield an Index.Basic
-- In NumPy there may only be a single ellipsis present.
inductive NumpyItem where
| int (n : Int)
| slice (slice : Slice)
| ellipsis
| newaxis
deriving BEq, Repr

abbrev NumpyBasic := List NumpyItem

/-
Slices do not subsume nat indexing. Slices always return an array
with the same dimension as the input, while a nat index always reduces the
dimensions by one.

# x
array([1, 2, 3, 4, 5, 6])

# x[0], x[0:1:1]
(np.int64(1), array([1]))

# x[0].shape, x[0:1:1].shape
((), (1,))
-/
abbrev BasicIter := List Slice.Iter

private def shapeToSliceIters (shape : List Nat) : Err (List Slice.Iter) :=
  shape.mapM fun dim => Slice.Iter.make Slice.all dim

-- Return the translated items, along with the new output shape
def toBasicIter (items : NumpyBasic) (shape : Shape) : Err (BasicIter × Shape) := do
  if 1 < items.count .ellipsis then .error "Index can contain at most one ellipsis"
  else if shape.val.length < items.length then .error "Too many indices"
  else
    let rec loop (items : NumpyBasic) (shape : List Nat) : Err (List Slice.Iter × List Nat) := match items, shape with
    -- If we have too few index items, the rest are assumed to be unchanged axes
    | [], shape => do
      let iters <- shapeToSliceIters shape
      .ok (iters, shape)
    | _, [] => .error "impossible" -- We checked above that there are at least as many dims as items
    | .int n :: items, dim :: shape => do
      -- constant indices throw on overflow/underflow
      if n <= -dim || dim <= n then .error "Constant index out of bounds" else
      -- Use a slice to do the finicky conversion to a position in the array
      -- Note that slices and numbers do not behave the same when negative
      -- e.g. np.arange(2)[-1] == 1 but np.arange(2)[-1:0:1] = []
      let n := if n < 0 then n + dim else n
      let slice := Slice.ofStartStop n (n+1)
      let iter <- Slice.Iter.make slice dim
      let (basic, shape) <- loop items shape
      -- Drop the dimension from the resulting shape
      .ok (iter :: basic, shape)
    | .slice slice :: items, dim :: shape => do
      let (basic, shape) <- loop items shape
      let slice <- Slice.Iter.make slice dim
      .ok (slice :: basic, slice.size :: shape)
    | .newaxis :: items, dim :: shape => do
      let (basic, shape) <- loop items shape
      let slice <- Slice.Iter.make Slice.all dim
      .ok (slice :: basic, dim :: shape)
    | .ellipsis :: items, dim :: shape => do
      if items.length == 1 + shape.length then
        loop items (dim :: shape)
      else  -- there are fewer items than the shape
        loop (.ellipsis :: items) shape
    let (items, dims) <- loop items shape.val
    return (items, Shape.mk dims)

def belist (iter : BasicIter) : Iterator.BEList Slice.Iter := Iterator.BEList.make iter

private def toList (basic : BasicIter) : List (List Nat) := Iterator.toList (belist basic)

#guard
  let slice := Slice.Iter.make! (Slice.ofInt 10) 11
  toList [slice, slice] == [[10, 10]]

#guard
  let slice := Slice.Iter.make! (Slice.ofInt 10) 12
  toList [slice, slice] == [[10, 10], [10, 11], [11, 10], [11, 11]]

#guard
  let slice := Slice.Iter.make! Slice.all 3
  toList [slice, slice] ==
    [[0, 0], [0, 1], [0, 2],
     [1, 0], [1, 1], [1, 2],
     [2, 0], [2, 1], [2, 2]]

def apply (index : NumpyBasic) (arr : Tensor) : Err Tensor := do
  let itemsize := arr.itemsize
  let oldShape := arr.shape
  let (basic, newShape) <- toBasicIter index oldShape
  let iter := belist basic
  let iterSize := Iterator.size (List Nat) (belist basic)
  let mut data := ByteArray.emptyWithCapacity (iterSize * itemsize)
  for dimIndex in iter do
    let posn := arr.dimIndexToPosition dimIndex
    for j in [0:itemsize] do
      let b := arr.data.get! (posn + j)
      data := data.push b
  return {
     dtype := arr.dtype,
     shape := newShape,
     data
  }

def apply! (index : NumpyBasic) (arr : Tensor) : Tensor := (get! (apply index arr))

/-
`arr[index] = v`

Since the dtypes need to be equal, for now, Element here is awkward. All we need it the byte size, which is in the Tensor
Refactoring would be good here.

Note that v is broadcast to arr[index].shape, but arr[index].shape is not broadcast to v, which wouldn't make sense.
E.g.

# x = np.arange(6).reshape(2, 3)
# x[1,:] = np.array([5,6,7])

is ok but

# x[1,:] = np.array([[1,2,3], [4,5,6]])

wouldn't make any sense, even though the shapes (1, 3) and (2, 3) are broadcastable.
-/
section Assign
open scoped Iterator.PairLockStep

def assign (arr : Tensor) (index : NumpyBasic) (v : Tensor) : Err Tensor := do
  let (basic, shape) <- toBasicIter index arr.shape
  let v <- v.broadcastTo shape
  let mut res := arr
  let aIter := belist basic
  let vIter := v.shape.belist
  if Iterator.size (List Nat) aIter != Iterator.size (List Nat) vIter then throw s!"Iterator size mismatch" else
  for (aIndex, vIndex) in (aIter, vIter) do
    let vVal <- v.getDimIndex vIndex
    res <- res.setDimIndex aIndex vVal
  return res

end Assign

def assign! (arr : Tensor) (index : NumpyBasic) (v : Tensor) : Tensor := get! $ assign arr index v

/-
For advanced indexing, the all-multidimensional-array case is relatively easy;
broadcast all arguments to the same shape, then select the elements of the original
array one by one. For example

# x = np.arange(6).reshape(2, 3)
# x
array([[0, 1, 2],
       [3, 4, 5]])
# i0 = np.array([1, 0])[:, None]
# ii = np.array([1, 2, 0])[None, :]
# x[i0, i1]
array([[4, 5, 3],
       [1, 2, 0]])

To obtain the later result, we simply walk through the [2, 3]-shaped indices
[[x[1, 1], x[1, 2], x[1, 0]],
 [x[0, 1], x[0, 2], x[0, 0]],

This also works when the dims of the index is smaller than the dims
of the array. Each x[i, j] is just an array instead of a scalar. We do not currently
implement that, but if we need it it will be clear what to do; we just copy the (contiguous)
bytes of the sub-array.

Mixing basic and advanced indexing is complex: https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing
While I can follow the individual examples, the general case is some work.
As a simple example of mixing, they give `x[..., ind, :]` where
`x.shape` is `(10, 20, 30)` and `ind` is a `(2, 5, 2)`-shaped indexing array.
The result has shape `(10, 2, 5, 2, 30)` and `result[..., i, j, k, :] = x[..., ind[i, j, k], :]`.
While this example is understandable, things get more complex, and I've not yet seen
examples we currently want to support that uses them. Therefore, we do not currently implement
mixed basic/advanced indexing.
-/

namespace Advanced

def apply (indexTensors : List Tensor) (arr : Tensor) : Err Tensor := do
  if indexTensors.any fun arr => !arr.isIntLike then .error "Index arrays must have an int-like type" else
  if indexTensors.length != arr.ndim then .error "advanced indexing length mismatch"
  -- Reshape all the input tesnsors
  match Broadcast.broadcastList (indexTensors.map fun arr => arr.shape) with
  | none => .error "input shapes must be broadcastable"
  | some outShape =>
  let mut reshapedIndexTensors := []
  for indexTensor in indexTensors do
    let indexTensor <- indexTensor.reshape outShape -- should never fail since broadcastList succeeds
    reshapedIndexTensors := indexTensor :: reshapedIndexTensors
  reshapedIndexTensors := reshapedIndexTensors.reverse
  let mut res := Tensor.zeros arr.dtype outShape
  -- Now we will iterate over the output shape, computing the values one-by-one from the input array
  for outDimIndex in outShape.belist do
    -- Get the index for each dimension in the original array from the corresponding value of the index tensors
    let mut inIntIndex : List Int := []
    for indexTensor in reshapedIndexTensors do
      let v <- indexTensor.intAtDimIndex outDimIndex
      inIntIndex := v :: inIntIndex
    let inDimIndex <- arr.shape.intIndexToDimIndex inIntIndex.reverse
    let bytes <- arr.byteArrayAtDimIndex inDimIndex
    res <- res.setByteArrayAtDimIndex outDimIndex bytes
  return res

def apply! (indexTensors : List Tensor) (arr : Tensor) : Tensor := get! $ apply indexTensors arr

end Advanced

section Test
open Tensor.Format.Tree

/-
0 1 2
3 4 5
6 7 8

9 10 11
12 12 14
15 16 17

18 19 20
21 22 23
24 25 26
-/
#guard
  let tp := Dtype.int8
  let ind0 := (Tensor.ofIntList! tp [1, 2, 0, 0]).reshape! (Shape.mk [2, 2])
  let ind1 := (Tensor.ofIntList! tp [2, -2, 0, 1]).reshape! (Shape.mk [2, 2])
  let ind2 := (Tensor.ofIntList! tp [1, 1, -1, -1]).reshape! (Shape.mk [2, 2])
  let arr := (Tensor.arange! Dtype.uint8 27).reshape! (Shape.mk [3, 3, 3])
  let res := Advanced.apply! [ind0, ind1, ind2] arr
  let tree := res.toNatTree!
  tree == Tensor.Format.Tree.node [.root [16, 22], .root [2, 5]]

#guard
  let tp := Dtype.int8
  let tensor := (Tensor.arange! tp 10).reshape! (Shape.mk [2, 5])
  let index := [NumpyItem.int 1]
  let res := get! $ apply index tensor
  let tree := res.toNatTree!
  let tree' := .root [5, 6, 7, 8, 9]
  tree == tree'

#guard
  let tp := Dtype.int8
  let tensor := (Tensor.arange! tp 10).reshape! $ Shape.mk [2, 5]
  let index := [.int 1]
  -- Bug in #guard keeps me from using `let (arr, copied) := ...` here
  let arr := apply! index tensor
  let tree' := .root [5, 6, 7, 8, 9]
  arr.toNatTree! == tree'

#guard
  let tp := Dtype.int8
  let tensor := (Tensor.arange! tp 20).reshape! $ Shape.mk [2, 2, 5]
  let index := [.int 1, .int 1, .int 4]
  let arr := apply! index tensor
  let tree := arr.toIntTree!
  let tree' := .root [19]
  tree == tree'

#guard
  let tp := Dtype.uint8
  let tensor1 := (Tensor.arange! tp 20).reshape! (Shape.mk [2, 2, 5])
  let index := [.int 1, .int 1, .int 4]
  let tensor2 := Tensor.arrayScalarNat! tp 255
  let res := assign! tensor1 index tensor2
  let tree := res.toNatTree!
  let tree' := node [
    node [ root [0, 1, 2, 3, 4], root [5, 6, 7, 8, 9] ],
    node [ root [10, 11, 12, 13, 14], root [15, 16, 17, 18, 255] ],
  ]
  tree == tree'

#guard
  let tp := Dtype.uint8
  let tensor1 := (Tensor.arange! tp 20).reshape! (Shape.mk [2, 2, 5])
  let index := [.int 1, .int 1, .newaxis]
  let tensor2 := Tensor.ofNatList! tp [50, 60, 70, 80, 90]
  let res := assign! tensor1 index tensor2
  let tree := res.toNatTree!
  let tree' := node [
    node [ root [0, 1, 2, 3, 4], root [5, 6, 7, 8, 9] ],
    node [ root [10, 11, 12, 13, 14], root [50, 60, 70, 80, 90] ],
  ]
  tree == tree'

#guard
  let tp := Dtype.uint8
  let tensor1 := (Tensor.arange! tp 20).reshape! (Shape.mk [2, 2, 5])
  let index := [.int 1, .int 1, .slice (Slice.ofStartStop 1 4)]
  let tensor2 := Tensor.ofNatList! tp [50, 60, 70]
  let res := assign! tensor1 index tensor2
  let tree := res.toNatTree!
  let tree' := node [
    node [ root [0, 1, 2, 3, 4], root [5, 6, 7, 8, 9] ],
    node [ root [10, 11, 12, 13, 14], root [15, 50, 60, 70, 19] ],
  ]
  tree == tree'

#guard
  let tp := Dtype.uint8
  let tensor1 := (Tensor.arange! tp 20).reshape! (Shape.mk [4, 5])
  let index := [.slice (Slice.ofStartStop 1 3), .slice (Slice.ofStartStop 1 4)]
  let tensor2 := (Tensor.ofNatList! tp [40, 50, 60, 70, 80, 90]).reshape! (Shape.mk [2, 3])
  let res := get! $ assign tensor1 index tensor2
  let tree := res.toNatTree!
  let tree' := node [
      root [0, 1, 2, 3, 4],
      root [5, 40, 50, 60, 9],
      root [10, 70, 80, 90, 14],
      root [15, 16, 17, 18, 19]
  ]
  tree == tree'

#guard
  let tp := Dtype.uint8
  let tensor1 := (Tensor.arange! tp 20).reshape! (Shape.mk [4, 5])
  let index := [NumpyItem.slice (Slice.ofStartStop 1 3), .slice (Slice.ofStartStop 1 4)]
  let tensor2 := Tensor.ofNatList! tp [40, 50, 60] -- tensor2 should be broadcast to (2, 3)
  let res := assign! tensor1 index tensor2
  let tree := res.toNatTree!
  let tree' := node [
      root [0, 1, 2, 3, 4],
      root [5, 40, 50, 60, 9],
      root [10, 40, 50, 60, 14],
      root [15, 16, 17, 18, 19]
  ]
  tree == tree'

private def numpyBasicToList (dims : List Nat) (basic : NumpyBasic) : Option (List (List Nat)) := do
  let shape := Shape.mk dims
  let (basic, _) <- (toBasicIter basic shape).toOption
  return toList basic

#guard numpyBasicToList [] [] == some [[]]
#guard numpyBasicToList [1] [.int 0] == some [[0]]
#guard numpyBasicToList [2] [.int 0] == some [[0]]
#guard numpyBasicToList [2] [.int 1] == some [[1]]
#guard numpyBasicToList [2] [.int 2] == none
#guard numpyBasicToList [2] [.int (-1)] == some [[1]]
#guard numpyBasicToList [2] [.int (-3)] == none
#guard numpyBasicToList [4] [.slice Slice.all] == some [[0], [1], [2], [3]]
#guard numpyBasicToList [4] [.slice $ Slice.make! .none .none (.some 2)] == some [[0], [2]]
#guard numpyBasicToList [4] [.slice $ Slice.make! (.some (-1)) .none (.some (-2))] == some [[3], [1]]
#guard numpyBasicToList [2, 2] [.int 5] == none
#guard numpyBasicToList [2, 2] [.int 0] == some [[0, 0], [0, 1]]
#guard numpyBasicToList [2, 2] [.int 0, .int 0] == some [[0, 0]]
#guard numpyBasicToList [2, 2] [.int 0, .int 1] == some [[0, 1]]
#guard numpyBasicToList [2, 2] [.int 0, .int 2] == none
#guard numpyBasicToList [3, 3] [.slice Slice.all, .int 2] == some [[0, 2], [1, 2], [2, 2]]
#guard numpyBasicToList [3, 3] [.int 2, .slice Slice.all] == some [[2, 0], [2, 1], [2, 2]]
#guard numpyBasicToList [2, 2] [.slice Slice.all, .slice Slice.all] == some [[0, 0], [0, 1], [1, 0], [1, 1]]
#guard numpyBasicToList [2, 2] [.slice (Slice.make! .none .none (.some (-1))), .slice Slice.all] == some [[1, 0], [1, 1], [0, 0], [0, 1]]
#guard numpyBasicToList [4, 2] [.slice (Slice.make! .none .none (.some (-2))), .slice Slice.all] == some [[3, 0], [3, 1], [1, 0], [1, 1]]

end Test

end Index
end TensorLib
