/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/

import TensorLib.Common
import TensorLib.Tensor
import TensorLib.Broadcast

/-!
Universal functions: https://numpy.org/doc/stable/reference/ufuncs.html
-/

namespace TensorLib
namespace Tensor
namespace Ufunc

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
  return Element.arrayScalar acc

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

private def hasTree0 (a : Type) [BEq a] [Element a] (arr : Tensor) (n : a) : Bool :=
  arr.shape.val == [] && match Element.getPosition arr 0 with
  | .error _ => false
  | .ok (v : a) => v == n

private def hasTree1 (a : Type) [Repr a] [BEq a] [Element a] (arr : Tensor) (xs : List a) : Bool :=
  arr.shape.val == [xs.length] && match arr.toTree a with
  | .error _ => false
  | .ok v => v == .root xs

#guard
  let typ := BV8
  let arr := get! $ (Element.arange typ 10).reshape (Shape.mk [2, 5])
  !(sum typ arr (.some [0, 1, 0])).isOk &&
  !(sum typ arr (.some [0, 0, 1])).isOk &&
  !(sum typ arr (.some [7])).isOk

-- [[0, 1, 2, 3, 4],
--  [5, 6, 7, 8, 9]]
#guard
  let typ := BV8
  let arr := get! $ (Element.arange typ 10).reshape (Shape.mk [2, 5])
  let x0 := get! $ sum typ arr .none
  let x1 := get! $ sum typ arr (.some [])
  let x2 := get! $ sum typ arr (.some [0])
  let x3 := get! $ sum typ arr (.some [1])
  let x4 := get! $ sum typ arr (.some [1, 0])
  let x5 := get! $ sum typ arr (.some [0, 1])
  let res :=
     hasTree0 typ x0 45 &&
     hasTree0 typ x1 45 &&
     hasTree1 typ x2 [5, 7, 9, 11, 13] &&
     hasTree1 typ x3 [10, 35] &&
     hasTree0 typ x4 45 &&
     hasTree0 typ x5 45
  res

#guard
  let typ := BV8
  let x := Element.arange typ 10
  let arr := get! $ add typ x x
  hasTree1 typ arr [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

#guard
  let typ := BV8
  let x := Element.arange typ 10
  let y := Element.arrayScalar (7 : typ)
  let arr := get! $ add typ x y
  hasTree1 typ arr [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

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

end Ufunc
