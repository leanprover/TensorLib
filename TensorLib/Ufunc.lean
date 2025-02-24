/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/

import TensorLib.Broadcast
import TensorLib.Common
import TensorLib.Index
import TensorLib.Tensor

/-!
Universal functions: https://numpy.org/doc/stable/reference/ufuncs.html
-/

namespace TensorLib
namespace Tensor
namespace Ufunc

def DEBUG : Bool := false

private def binop (a : Type) [Element a] (x y : Tensor) (op : a -> a -> Err a) : Err Tensor :=
  match Broadcast.broadcast { left := x.shape, right := y.shape } with
  | .none => .error s!"Can't broadcast shapes ${x.shape} with {y.shape}"
  | .some shape =>
    if x.dtype != y.dtype then .error s!"Casting between dtypes is not implemented yet: {repr x.dtype} <> {repr y.dtype}" else
    do
      let mut arr := Tensor.empty x.dtype shape
      let iter := DimsIter.make shape
      for idx in iter do
        let v <- Element.getDimIndex x idx
        let w <- Element.getDimIndex y idx
        let k <- op v w
        let arr' <- Element.setDimIndex arr idx k
        arr := arr'
      .ok arr

def add (a : Type) [Add a] [Element a] (x y : Tensor) : Err Tensor :=
  binop a x y (fun x y => .ok (x + y))

def sub (a : Type) [Sub a] [Element a] (x y : Tensor) : Err Tensor :=
  binop a x y (fun x y => .ok (x - y))

def mul (a : Type) [Mul a] [Element a] (x y : Tensor) : Err Tensor :=
  binop a x y (fun x y => .ok (x * y))

def div (a : Type) [Div a] [Element a] (x y : Tensor) : Err Tensor :=
  binop a x y (fun x y => .ok (x / y))

/-
TODO:
- np.sum. Prove that np.sum(x, axis=(2, 4, 6)) == np.sum(np.sum(np.sum(x, axis=6), axis=4), axis=2) # and other variations
-/

-- Sum with no axis. Adds all the elements.
private def sum0 (a : Type) [Add a] [Zero a] [Element a] (arr : Tensor) : Err Tensor := do
  let mut acc : a := 0
  let mut iter := DimsIter.make arr.shape
  for index in iter do
    let n : a <- Element.getDimIndex arr index
    acc := Add.add acc n
  return Element.arrayScalar a acc

-- Sum with a single axis.
private def sum1 (a : Type) [Add a] [Zero a] [Element a] (arr : Tensor) (axis : Nat) : Err Tensor := do
  if arr.ndim <= axis then .error "axis out of range" else
  let oldshape := arr.shape
  let (leftShape, rightShape) := oldshape.val.splitAt axis
  match rightShape with
  | [] => .error "Invariant failure"
  | dim :: dims =>
    let newshape := Shape.mk $ leftShape ++ dims
    let mut res := Tensor.zeros arr.dtype newshape
    for index in DimsIter.make newshape do
      let mut acc : a := 0
      for i in [0:dim] do
        let index' := index.insertIdx axis i
        let v : a <- Element.getDimIndex arr index'
        acc := acc + v
      res <- Element.setDimIndex res index acc
    return res

-- Remove duplicate elements in a sorted list
private def uniq [BEq a] (xs : List a) : Bool := match xs with
| [] | [_] => true
| x1 :: x2 :: xs => x1 != x2 && uniq (x2 :: xs)

def sum (a : Type) [Add a] [Zero a] [Element a] (arr : Tensor) (axes : Option (List Nat)) : Err Tensor :=
  match axes with
  | .none => sum0 a arr
  | .some axes =>
  let axes := (List.mergeSort axes).reverse
  if !(uniq axes) then .error "Duplicate axis elements" else
  match axes with
  | [] => sum0 a arr
  | axis :: axes => do
    let mut res <- sum1 a arr axis
    let rec loop (axes : List Nat) (acc : Tensor) : Err Tensor := match axes with
    | [] => .ok acc
    | axis :: axes => do
      let acc <- sum1 a acc axis
      let axes := axes.map fun n => n-1 -- When we remove an axis, all later axes point to one dimension less
      loop axes acc
    termination_by axes.length
    loop axes res

/-
Implements the dot product. np.dot for 1-D arrays.
np.dot supports a bunch of other cases, but all of them are reducible to other operations like
multiplication by a scalar, matrix multiplication, etc. While we'd like to stay close to NumPy,
we also would like the author to use the simplest, most natural operations possible.
-/
def dot (a : Type) [Add a] [Mul a] [Zero a] [Element a] (x y : Tensor) : Err Tensor := do
  if x.dtype != y.dtype then .error "Expected same dtype" else
  let (xd1, yd1) <- match x.shape.val, y.shape.val with
  | [xd1], [yd1] => .ok (xd1, yd1)
  | [], _ | _, [] => .error "While allowed in NumPy, please use scalar multiplication for array scalars"
  | _, _ => .error "While allowed in NumPy when the dimensions work out, please use matmul for this use case"
  if xd1 != yd1 then .error "dot: reduction dimension mismatch" else
  let mut acc : a := 0
  for i in [0:xd1] do
    let u <- Element.getDimIndex x [i]
    let v <- Element.getDimIndex y [i]
    acc := acc + u * v
  return Element.arrayScalar a acc

-- The usual 2D matmul
private def matmul2 (a : Type) [Add a] [Mul a] [Zero a] [Element a] (x y : Tensor) : Err Tensor := do
  if x.dtype != y.dtype then .error "Expected same dtype" else
  let (xd1, xd2, yd1, yd2) <- match x.shape.val, y.shape.val with
  | [xd1, xd2], [yd1, yd2] => .ok (xd1, xd2, yd1, yd2)
  | _, _ => .error "Expected 2d arrays"
  if xd2 != yd1 then .error "matmul2: reduction dimension mismatch" else
  let mut res := Tensor.zeros x.dtype (Shape.mk [xd1, yd2])
  for i in [0:xd1] do
    for j in [0:yd2] do
      let mut acc : a := 0
      for k in [0:xd2] do
        let u <- Element.getDimIndex x [i, k]
        let v <- Element.getDimIndex y [k, j]
        acc := acc + u * v
      res <- Element.setDimIndex res [i, j] acc
  return res

/-
NumPy matmul handles many variants of dimensions.
https://numpy.org/doc/2.1/reference/generated/numpy.matmul.html
For now we'll handle
* 2x2
* NxM where N,M >= 2, where when N or M are greater than 2, we just have a lot of 2x2 matrics
  that we multiply together and put in the result in the correctly-shaped slots.

TODO: I'm not sure what to do with the axis/axes arguments.
-/
def matmul (a : Type) [Add a] [Mul a] [Zero a] [Element a] (x y : Tensor) : Err Tensor := do
  if x.dtype != y.dtype then .error "Expected same dtype" else
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
  | some (Shape.mk []) => matmul2 a x y
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
    let (x', _) <- Index.apply index x
    let (y', _) <- Index.apply index y
    let v <- matmul2 a x' y'
    res <- Index.assign a res index v
  -- now reshape
  let resShape := prefixShape.append [xd1, yd2]
  res.reshape resShape

section Test
open Tensor.Format.Tree


/-
# x = np.arange(10).reshape(2, 5)
# y = np.arange(10).reshape(5, 2)
# np.matmul(x, y)
array([[ 60,  70],
       [160, 195]])
-/
#guard
  let tp := BV8
  let arr1 := get! $ (Element.arange tp 10).reshape (Shape.mk [2, 5])
  let arr2 := get! $ (Element.arange tp 10).reshape (Shape.mk [5, 2])
  let arr3 := get! $ matmul2 tp arr1 arr2
  arr3.toTree! tp == .node [.root [60, 70], .root [160, 195]]

#guard
  let tp := BV8
  let arr := get! $ (Element.arange tp 10).reshape (Shape.mk [2, 5])
  !(sum tp arr (.some [0, 1, 0])).isOk &&
  !(sum tp arr (.some [0, 0, 1])).isOk &&
  !(sum tp arr (.some [7])).isOk

-- [[0, 1, 2, 3, 4],
--  [5, 6, 7, 8, 9]]
#guard
  let tp := BV8
  let arr := get! $ (Element.arange tp 10).reshape (Shape.mk [2, 5])
  let x0 := get! $ sum tp arr .none
  let x1 := get! $ sum tp arr (.some [])
  let x2 := get! $ sum tp arr (.some [0])
  let x3 := get! $ sum tp arr (.some [1])
  let x4 := get! $ sum tp arr (.some [1, 0])
  let x5 := get! $ sum tp arr (.some [0, 1])
  x0.toTree! tp == .root [45]
  && x1.toTree! tp == .root [45]
  && x2.toTree! tp == .root [5, 7, 9, 11, 13]
  && x3.toTree! tp == .root [10, 35]
  && x4.toTree! tp == .root [45]
  && x5.toTree! tp == .root [45]

#guard
  let tp := BV8
  let x := Element.arange tp 10
  let arr := get! $ add tp x x
  arr.toTree! tp == .root [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

#guard
  let tp := BV8
  let x := Element.arange tp 10
  let y := Element.arrayScalar tp 7
  let arr := get! $ add tp x y
  arr.toTree! tp == .root [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

#guard
  let tp := BV8
  let x := (Element.arange tp 6).reshape! (Shape.mk [2, 3])
  let y := (Element.arange tp 6).reshape! (Shape.mk [3, 2])
  let z := get! $ matmul tp x y
  z.toTree! tp == .node [.root [10, 13], .root [28, 40]]

#guard
  let tp := BV8
  let x := (Element.arange tp 6).reshape! (Shape.mk [1, 2, 3])
  let y := (Element.arange tp 6).reshape! (Shape.mk [1, 3, 2])
  let z := get! $ matmul tp x y
  z.toTree! tp == .node [.node [.root [10, 13], .root [28, 40]]]

#guard
  let tp := BV8
  let x := (Element.arange tp 12).reshape! (Shape.mk [2, 2, 3])
  let y := (Element.arange tp 6).reshape! (Shape.mk [1, 3, 2])
  let z := get! $ matmul tp x y
  z.toTree! tp == .node [
    .node [.root [10, 13], .root [28, 40]],
    .node [.root [46, 67], .root [64, 94]]
  ]

#guard
  let tp := BV8
  let x := (Element.arange tp 12).reshape! (Shape.mk [2, 1, 2, 3])
  let y := (Element.arange tp 6).reshape! (Shape.mk [3, 2])
  let z := get! $ matmul tp x y
  z.toTree! tp == .node [
    .node [.node [.root [10, 13], .root [28, 40]]],
    .node [.node [.root [46, 67], .root [64, 94]]]
  ]

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
