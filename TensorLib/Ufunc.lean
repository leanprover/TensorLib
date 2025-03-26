/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/

import TensorLib.Broadcast
import TensorLib.Common
import TensorLib.Dtype
import TensorLib.Index
import TensorLib.Iterator
import TensorLib.Tensor

/-!
Universal functions: https://numpy.org/doc/stable/reference/ufuncs.html
-/

namespace TensorLib
namespace Tensor
namespace Ufunc

def DEBUG : Bool := false

def unop (x : Tensor) (op : ByteArray -> Err ByteArray) : Err Tensor := do
  let mut arr := Tensor.empty x.dtype x.shape
  for idx in x.shape.belist do
    let v <- x.getDimIndex idx
    let k <- op v
    let arr' <- arr.setDimIndex idx k
    arr := arr'
  return arr

def binop (x y : Tensor) (op : ByteArray -> ByteArray -> Err ByteArray) : Err Tensor :=
  if x.dtype != y.dtype then .error "Implicit type conversions are not supported" else
  match Broadcast.broadcast { left := x.shape, right := y.shape } with
  | .none => .error s!"Can't broadcast shapes: ${x.shape} {y.shape}"
  | .some shape => do
    let x <- x.broadcastTo shape
    let y <- y.broadcastTo shape
    let mut arr := Tensor.empty x.dtype shape
    for idx in shape.belist do
      let v <- x.getDimIndex idx
      let w <- y.getDimIndex idx
      let k <- op v w
      let arr' <- arr.setDimIndex idx k
      arr := arr'
    return arr

def add (x y : Tensor) : Err Tensor := binop x y x.dtype.add
def sub (x y : Tensor) : Err Tensor := binop x y x.dtype.sub
def mul (x y : Tensor) : Err Tensor := binop x y x.dtype.mul
def div (x y : Tensor) : Err Tensor := binop x y x.dtype.div

def add! (x y : Tensor) : Tensor := get! $ add x y
def sub! (x y : Tensor) : Tensor := get! $ sub x y
def mul! (x y : Tensor) : Tensor := get! $ mul x y
def div! (x y : Tensor) : Tensor := get! $ div x y

def abs (x : Tensor) : Err Tensor := unop x x.dtype.abs
def abs! (x : Tensor) : Tensor := get! $ abs x

/-
TODO:
- np.sum. Prove that np.sum(x, axis=(2, 4, 6)) == np.sum(np.sum(np.sum(x, axis=6), axis=4), axis=2) # and other variations
-/

-- Sum with no axis. Adds all the elements.
private def sum0 (arr : Tensor) : Err Tensor := do
  let dtype := arr.dtype
  let mut acc := dtype.zero
  for index in arr.shape.belist do
    let n <- arr.getDimIndex index
    let acc' <- dtype.add acc n
    acc := acc'
  Tensor.arrayScalar dtype acc

-- Sum with a single axis.
private def sum1 (arr : Tensor) (axis : Nat) : Err Tensor := do
  let dtype := arr.dtype
  if arr.ndim <= axis then .error "axis out of range" else
  let oldshape := arr.shape
  let (leftShape, rightShape) := oldshape.val.splitAt axis
  match rightShape with
  | [] => .error "Invariant failure"
  | dim :: dims =>
    let newshape := Shape.mk $ leftShape ++ dims
    let mut res := Tensor.zeros arr.dtype newshape
    for index in newshape.belist do
      let mut acc := dtype.zero
      for i in [0:dim] do
        let index' := index.insertIdx axis i
        let v <- arr.getDimIndex index'
        let acc' <- dtype.add acc v
        acc := acc'
      res <- res.setDimIndex index acc
    return res

-- Remove duplicate elements in a sorted list
private def uniq [BEq a] (xs : List a) : Bool := match xs with
| [] | [_] => true
| x1 :: x2 :: xs => x1 != x2 && uniq (x2 :: xs)

def sum (arr : Tensor) (axes : Option (List Nat)) : Err Tensor :=
  match axes with
  | .none => sum0 arr
  | .some axes =>
  let axes := (List.mergeSort axes).reverse
  if !(uniq axes) then .error "Duplicate axis elements" else
  match axes with
  | [] => sum0 arr
  | axis :: axes => do
    let mut res <- sum1 arr axis
    let rec loop (axes : List Nat) (acc : Tensor) : Err Tensor := match axes with
    | [] => .ok acc
    | axis :: axes => do
      let acc <- sum1 acc axis
      let axes := axes.map fun n => n-1 -- When we remove an axis, all later axes point to one dimension less
      loop axes acc
    termination_by axes.length
    loop axes res

def sum! (arr : Tensor) (axes : Option (List Nat)) : Tensor := get! $ sum arr axes

/-
Implements the dot product. np.dot for 1-D arrays.
np.dot supports a bunch of other cases, but all of them are reducible to other operations like
multiplication by a scalar, matrix multiplication, etc. While we'd like to stay close to NumPy,
we also would like the author to use the simplest, most natural operations possible.
-/
def dot (x y : Tensor) : Err Tensor := do
  let dtype := x.dtype
  if dtype != y.dtype then .error "Expected same dtype" else
  let (xd1, yd1) <- match x.shape.val, y.shape.val with
  | [xd1], [yd1] => .ok (xd1, yd1)
  | [], _ | _, [] => .error "While allowed in NumPy, please use scalar multiplication for array scalars"
  | _, _ => .error "While allowed in NumPy when the dimensions work out, please use matmul for this use case"
  if xd1 != yd1 then .error "dot: reduction dimension mismatch" else
  let mut acc := dtype.zero
  for i in [0:xd1] do
    let u <- x.getDimIndex [i]
    let v <- y.getDimIndex [i]
    let m <- dtype.mul u v
    let acc' <- dtype.add acc m
    acc := acc'
  Tensor.arrayScalar dtype acc

-- The usual 2D matmul
private def matmul2 (x y : Tensor) : Err Tensor := do
  let dtype := x.dtype
  if dtype != y.dtype then .error "Expected same dtype" else
  let (xd1, xd2, yd1, yd2) <- match x.shape.val, y.shape.val with
  | [xd1, xd2], [yd1, yd2] => .ok (xd1, xd2, yd1, yd2)
  | _, _ => .error "Expected 2d arrays"
  if xd2 != yd1 then .error "matmul2: reduction dimension mismatch" else
  let mut res := Tensor.zeros x.dtype (Shape.mk [xd1, yd2])
  for i in [0:xd1] do
    for j in [0:yd2] do
      let mut acc := dtype.zero
      for k in [0:xd2] do
        let u <- x.getDimIndex [i, k]
        let v <- y.getDimIndex [k, j]
        let m <- dtype.mul u v
        let acc' <- dtype.add acc m
        acc := acc'
      res <- res.setDimIndex [i, j] acc
  return res

private def matmul2! (x y : Tensor) : Tensor := get! $ matmul2 x y

/-
NumPy matmul handles many variants of dimensions.
https://numpy.org/doc/2.1/reference/generated/numpy.matmul.html
For now we'll handle
* 2x2
* NxM where N,M >= 2, where when N or M are greater than 2, we just have a lot of 2x2 matrics
  that we multiply together and put in the result in the correctly-shaped slots.

TODO: I'm not sure what to do with the axis/axes arguments.
-/
def matmul (x y : Tensor) : Err Tensor := do
  let dtype := x.dtype
  if dtype != y.dtype then .error "Expected same dtype" else
  -- The last two dimensions of each array must line up matmul-style
  let (xd1, xd2, xds, yd1, yd2, yds) <- match x.shape.val.reverse, y.shape.val.reverse with
  | [], _ | _, [] => .error "array scalars not allowed"
  | [_], _ | _, [_] => .error "NumPy-like 1D matmul not yet implemented"
  | xd2 :: xd1 :: xds, yd2 :: yd1 :: yds => .ok (xd1, xd2, xds.reverse, yd1, yd2, yds.reverse)
  if xd2 != yd1 then .error "matmulN: reduction dimension mismatch" else
  -- Broadcast the prefixes (not including the final 2 dimensions, which wouldn't match under
  -- typical broadcast rules, (e.g. [4,2] vs [2,3] to produce a [4,3] matrix.)
  match Broadcast.broadcast { left := Shape.mk xds, right := Shape.mk yds } with
  | none => .error "can't broadcast prefix"
  | some (Shape.mk []) => matmul2 x y
  | some prefixShape =>
  -- First broadcast to get the correct sizes and strides ...
  let xShape := prefixShape.append [xd1, xd2]
  let yShape := prefixShape.append [yd1, yd2]
  let x <- x.broadcastTo xShape
  let y <- y.broadcastTo yShape
  -- then flatten
  let prefixSize := prefixShape.count
  let xShape := Shape.mk [prefixSize, xd1, xd2]
  let yShape := Shape.mk [prefixSize, yd1, yd2]
  let x <- x.reshape xShape
  let y <- y.reshape yShape
  -- then loop
  let resShape := Shape.mk [prefixSize, xd1, yd2]
  let mut res := Tensor.zeros x.dtype resShape
  for i in [0:prefixSize] do
    let index := [Index.NumpyItem.int i]
    let x' <- Index.apply index x
    let y' <- Index.apply index y
    let v <- matmul2 x' y'
    res <- Index.assign res index v
  -- now reshape
  let resShape := prefixShape.append [xd1, yd2]
  res.reshape resShape

private def matmul! (x y : Tensor) : Tensor := get! $ matmul x y

section Test
open Tensor.Format.Tree

private def lshift3 (arr : ByteArray) : Err ByteArray := match arr.data with
| #[byte] => return ByteArray.mk #[byte <<< 3]
| _ => throw "too many bytes"

#guard
  let tp := Dtype.uint8
  let arr1 := (Tensor.arange! tp 10).reshape! (Shape.mk [2, 5])
  let arr2 := get! $ unop arr1 lshift3
  let t : Format.Tree Nat := .node [.root [0, 1, 2, 3, 4], .root [5, 6, 7, 8, 9]]
  arr2.toNatTree! == t.map fun n => n * 8

/-
# x = np.arange(10).reshape(2, 5)
# y = np.arange(10).reshape(5, 2)
# np.matmul(x, y)
array([[ 60,  70],
       [160, 195]])
-/
#guard
  let tp := Dtype.uint8
  let arr1 := (Tensor.arange! tp 10).reshape! (Shape.mk [2, 5])
  let arr2 := (Tensor.arange! tp 10).reshape! (Shape.mk [5, 2])
  let arr3 := matmul2! arr1 arr2
  arr3.toNatTree! == .node [.root [60, 70], .root [160, 195]]

#guard
  let tp := Dtype.uint8
  let arr := (Tensor.arange! tp 10).reshape! (Shape.mk [2, 5])
  !(sum arr (.some [0, 1, 0])).isOk &&
  !(sum arr (.some [0, 0, 1])).isOk &&
  !(sum arr (.some [7])).isOk

/-
[[0, 1, 2, 3, 4],
 [5, 6, 7, 8, 9]]
-/
#guard
  let tp := Dtype.uint8
  let arr := (Tensor.arange! tp 10).reshape! (Shape.mk [2, 5])
  let x0 := sum! arr .none
  let x1 := sum! arr (.some [])
  let x2 := sum! arr (.some [0])
  let x3 := sum! arr (.some [1])
  let x4 := sum! arr (.some [1, 0])
  let x5 := sum! arr (.some [0, 1])
  x0.toNatTree! == .root [45]
  && x1.toNatTree! == .root [45]
  && x2.toNatTree! == .root [5, 7, 9, 11, 13]
  && x3.toNatTree! == .root [10, 35]
  && x4.toNatTree! == .root [45]
  && x5.toNatTree! == .root [45]

#guard
  let tp := Dtype.uint8
  let x := Tensor.arange! tp 10
  let arr := add! x x
  arr.toNatTree! == .root [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

#guard
  let tp := Dtype.uint8
  let x := Tensor.arange! tp 10
  let y := Tensor.arrayScalarNat! tp 7
  let arr := add! x y
  arr.toNatTree! == .root [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

#guard
  let tp := Dtype.uint8
  let x := (Tensor.arange! tp 6).reshape! (Shape.mk [2, 3])
  let y := (Tensor.arange! tp 6).reshape! (Shape.mk [3, 2])
  let z := matmul! x y
  z.toNatTree! == .node [.root [10, 13], .root [28, 40]]

#guard
  let tp := Dtype.uint8
  let x := (Tensor.arange! tp 6).reshape! (Shape.mk [1, 2, 3])
  let y := (Tensor.arange! tp 6).reshape! (Shape.mk [1, 3, 2])
  let z := matmul! x y
  z.toNatTree! == .node [.node [.root [10, 13], .root [28, 40]]]

#guard
  let tp := Dtype.uint8
  let x := (Tensor.arange! tp 12).reshape! (Shape.mk [2, 2, 3])
  let y := (Tensor.arange! tp 6).reshape! (Shape.mk [1, 3, 2])
  let z := matmul! x y
  z.toNatTree! == .node [
    .node [.root [10, 13], .root [28, 40]],
    .node [.root [46, 67], .root [64, 94]]
  ]

#guard
  let tp := Dtype.uint8
  let x := (Tensor.arange! tp 12).reshape! (Shape.mk [2, 1, 2, 3])
  let y := (Tensor.arange! tp 6).reshape! (Shape.mk [3, 2])
  let z := matmul! x y
  z.toNatTree! == .node [
    .node [.node [.root [10, 13], .root [28, 40]]],
    .node [.node [.root [46, 67], .root [64, 94]]]
  ]

#guard
  let shape := Shape.mk [2, 3]
  let t := (arange! Dtype.int8 6).reshape! shape
  let t := mul! (arrayScalarInt! Dtype.int8 (-1)) t
  let t1 := Tensor.ofNatList! Dtype.uint8 [0, 0xFF, 0xFE, 0xFD, 0xFC, 0xFB]
  let t1 := t1.astype! Dtype.int8
  let t1 := t1.reshape! shape
  Tensor.arrayEqual t t1

#guard
  let shape := Shape.mk [2, 3]
  let t := (arange! Dtype.int8 6).reshape! shape
  let t := t.astype! Dtype.float32
  let t1 := mul! t (Tensor.arrayScalarFloat32! (-1.0))
  let t1 := abs! t1
  Tensor.arrayEqual t t1

/-! WIP example NKI kernel
"""
NKI kernel to compute element-wise addition of two input tensors

This kernel assumes strict input/output tile-sizes, of up-to [128,512]

Args:
    a_input: a first input tensor, of shape [128,512]
    b_input: a second input tensor, of shape [128,512]
    c_output: an output tensor, of shape [128,512]
"""
private def nki_tensor_add_kernel_ (program_id0 program_id1 : Nat) (a_input b_input c_input : NumpyRepr) : Err Unit := do
  let tp := BV64

  -- Calculate tile offsets based on current 'program'
  let offset_i_x : tp := program_id0 * 128
  let offset_i_y : tp := program_id1 * 512
  -- Generate tensor indices to index tensors a and b
  let rx0 := Element.arange tp 128
  let rx <- rx0.reshape [128, 1]
  let ox := Element.arrayScalar offset_i_x
  let ix <- Ufunc.add tp ox rx
  let ry0 := Element.arange tp 128
  let ry <- ry0.reshape [1, 512]
  let oy := Element.arrayScalar offset_i_y
  let iy <- Ufunc.add tp oy ry
  let a_tile <- sorry -- load from a_input
  let b_tile <- sorry -- load from b_input
  let c_tile <- Ufunc.add tp a_tile b_tile
  let () <- sorry -- store to c_input
  .ok ()
-/
end Test
end Ufunc
end Tensor
end TensorLib
