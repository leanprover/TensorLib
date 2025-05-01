/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/

import Plausible

namespace TensorLib

export Function(comp)

abbrev Err := Except String

def impossible {a : Type} [h : Inhabited a] (msg : String := "") := @panic a h s!"Invariant violation: {msg}"

def get! [Inhabited a] (x : Err a) : a := match x with
| .error msg => impossible msg
| .ok x => x

def natDivCeil (num denom : Nat) : Nat := (num + denom - 1) / denom

instance [BEq a] : BEq (Err a) where
  beq x y := match x, y with
  | .ok x, .ok y => x == y
  | .error x, .error y => x == y
  | _, _ => false

def dot [Add a][Mul a][Zero a] (x y : List a) : a := (x.zip y).foldl (fun acc (a, b) => acc + a * b) 0

section Test
open Plausible

/--
info: Unable to find a counter-example
---
warning: declaration uses 'sorry'
-/
#guard_msgs in
example (x y : Nat) :
  let c := natDivCeil x y
  let f := x / y
  c == f || c == (f + 1) := by plausible

local instance : SampleableExt (Nat × Nat) :=
  SampleableExt.mkSelfContained do
    let x ← SampleableExt.interpSample Nat
    let n <- SampleableExt.interpSample Nat
    return (x * n, x)

/--
info: Unable to find a counter-example
---
warning: declaration uses 'sorry'
-/
#guard_msgs in
example (xy : Nat × Nat) :
  let (x, y) := xy
  x % y = 0 → x / y = natDivCeil x y := by
  plausible

end Test

def natProd (shape : List Nat) : Nat := shape.foldl (fun x y => x * y) 1

/-!
The strides are how many bytes you need to skip to get to the next element in that
"row". For example, in an array of 8-byte data with shape 2, 3, the strides are (24, 8).
Strides are also useful assuming 1-byte data for going back and forth between different
representations of the data. E.g. see `Tree` below.

Strides is typically used with the dtype size. However, it is also useful with byte size
1 to convert between different representations of the data. We call strides of the later
form "unit strides". So for example, a 1D array of f32 with normal indexing would have
a stride of 4 and a unit stride of 1. An f16 would have stride 2, unit stride 1.
-/
abbrev Strides := List Int

-- The term "index" is overloaded a lot in NumPy. We don't have great alternative names,
-- so we're going with different extensions of the term. The `DimIndex` is the index in a
-- multi-dimensional array (as opposed to in the flat data, see `Position` below).
-- For example, in an array with shape (4, 2, 3), the `DimIndex` of the last element is (3, 1, 2)
-- DimIndex is independent of the size of the dtype.
abbrev DimIndex := List Nat

/-!
A `Position` is the index in the flat data array. For example, In an array with shape
2, 3, the position of the element at (1, 2) is 5.
Position is independent of the size of the dtype.
-/
abbrev Position := Nat

/-!
A `Offset` is the positional offset from the starting position. For negative values, the magnitude
should be smaller than the starting position. So for tensor `t`, `0 <= t.startingPosition + offset < t.count`.
-/
abbrev Offset := Int

end TensorLib
