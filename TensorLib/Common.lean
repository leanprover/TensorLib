/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/

import Plausible
import Std.Tactic.BVDecide
import TensorLib.Iterator

open Plausible

namespace TensorLib

private def plausibleDefaultConfig : Plausible.Configuration := {}
/-
This suppresses messages from Plausible on the commandline when building, such as
    info: ././././TensorLib/Common.lean:37:0: Unable to find a counter-example
If you want to see the counterexample in the IDE, you need to remove the configuration argument,
but remember to add it back once you've fixed the bug.
-/
def cfg : Plausible.Configuration := { plausibleDefaultConfig with quiet := true }

instance [BEq a] : BEq (Err a) where
  beq x y := match x, y with
  | .ok x, .ok y => x == y
  | .error x, .error y => x == y
  | _, _ => false

instance : BEq ByteArray where
  beq x y := x.data == y.data

def dot [Add a][Mul a][Zero a] (x y : List a) : a := (x.zip y).foldl (fun acc (a, b) => acc + a * b) 0

section Test

#eval Testable.check (cfg := cfg) (
  ∀ (x y : Nat),
  let c := natDivCeil x y
  let f := x / y
  c == f || c == (f + 1)
)

local instance : SampleableExt (Nat × Nat) :=
  SampleableExt.mkSelfContained do
    let x ← SampleableExt.interpSample Nat
    let n <- SampleableExt.interpSample Nat
    return (x * n, x)

#eval Testable.check (cfg := cfg) (
  ∀ (xy : Nat × Nat),
  let (x, y) := xy
  x % y = 0 → x / y = natDivCeil x y
)

end Test

def natProd (shape : List Nat) : Nat := shape.foldl (fun x y => x * y) 1

-- We generally have large tensors, so don't show them by default
instance ByteArrayRepr : Repr ByteArray where
  reprPrec x _ :=
    if x.size < 100 then x.toList.repr 100 else
    s!"ByteArray of size {x.size}"

def _root_.ByteArray.reverse (arr : ByteArray) : ByteArray := ⟨ arr.data.reverse ⟩

def _root_.ByteArray.toNat (arr : ByteArray) : Nat := Id.run do
  let mut n : Nat := 0
  let nbytes := arr.size
  for i in [0:nbytes] do
    let v : UInt8 := arr.get! i
    n := n + Pow.pow 2 (8 * i) * v.toNat
  return n

#guard (ByteArray.mk #[1, 1]).toNat == 257
#guard (ByteArray.mk #[0, 1]).toNat == 256
#guard (ByteArray.mk #[0xFF, 0xFF]).toNat == 65535
#guard (ByteArray.mk #[0, 0x80]).toNat == 32768
#guard (ByteArray.mk #[0x80, 0]).toNat == 0x80

def _root_.ByteArray.toInt (arr : ByteArray) : Int := Id.run do
  let mut n : Nat := 0
  let nbytes := arr.size
  let signByte := arr.get! (nbytes - 1)
  let negative := 128 <= signByte
  for i in [0:nbytes] do
    let v : UInt8 := arr.get! i
    let v := if negative then UInt8.complement v else v
    n := n + Pow.pow 2 (8 * i) * v.toNat
  return if 128 <= signByte then -(n + 1) else n

#guard (ByteArray.mk #[1, 1]).toInt == 257
#guard (ByteArray.mk #[0, 1]).toInt == 256
#guard (ByteArray.mk #[1, 0]).toInt == 1
#guard (ByteArray.mk #[0xFF, 0xFF]).toInt == -1
#guard (ByteArray.mk #[0, 0x80]).toInt == -32768
#guard (ByteArray.mk #[0x80, 0]).toInt == 0x80

def bitVecToByteArray (n : Nat) (v : BitVec n) : ByteArray := Id.run do
  let numBytes := natDivCeil n 8
  let mut arr := ByteArray.mkEmpty numBytes
  for i in [0 : numBytes] do
    let byte := (v.ushiftRight (i * 8) &&& 0xFF).toNat.toUInt8
    arr := arr.push byte
  return arr

#guard (bitVecToByteArray 16 0x0100).toList == [0, 1]
#guard (bitVecToByteArray 20 0x01000).toList == [0, 16, 0]
#guard (bitVecToByteArray 32 0x1).toList == [1, 0, 0, 0]

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

abbrev BV8 := BitVec 8

def BV8.ofNat (i : Nat) : BV8 := i.toUInt8.toBitVec

-- `_root_` is required to add dot methods to UInt8, which is outside TensorLib
def _root_.UInt8.toBV8 (n : UInt8) : BV8 := BitVec.ofFin n.val
def BV8.toUInt8 (n : BV8) : UInt8 := UInt8.ofNat n.toFin

def BV8.toByteArray (x : BV8) : ByteArray := bitVecToByteArray 8 x

def _root_.ByteArray.toBV8 (x : ByteArray) (startIndex : Nat) : Err BV8 :=
  let n := startIndex
  if H7 : n < x.size then
    let H0 : n + 0 < x.size := by omega
    let x0 := x.get (Fin.mk _ H0)
    .ok (UInt8.toBV8 x0)
  else .error s!"Index out of range: {n}"

abbrev BV16 := BitVec 16

def BV16.toByteArray (x : BV16) : ByteArray := bitVecToByteArray 16 x


def BV16.toBytes (n : BV16) : BV8 × BV8 :=
  let n0 := (n >>> 0o00 &&& 0xFF).truncate 8
  let n1 := (n >>> 0o10 &&& 0xFF).truncate 8
  (n0, n1)

def BV16.ofBytes (x0 x1 : BV8) : BV16 :=
  (x0.zeroExtend 16 <<< 0o00) |||
  (x1.zeroExtend 16 <<< 0o10)

theorem BV16.BytesRoundTrip (n : BV16) :
  let (x0, x1) := BV16.toBytes n
  let n' := BV16.ofBytes x0 x1
  n = n' := by
    unfold BV16.toBytes BV16.ofBytes
    bv_decide

theorem BV16.BytesRoundTrip1 (x0 x1 : BV8) :
  let n := BV16.ofBytes x0 x1
  let (x0', x1') := BV16.toBytes n
  x0 = x0' &&
  x1 = x1' := by
    unfold BV16.toBytes BV16.ofBytes
    bv_decide

def _root_.ByteArray.toBV16 (x : ByteArray) (startIndex : Nat) : Err BV16 :=
  let n := startIndex
  if H7 : n + 1 < x.size then
    let H0 : n + 0 < x.size := by omega
    let H1 : n + 1 < x.size := by omega
    let x0 := x.get (Fin.mk _ H0)
    let x1 := x.get (Fin.mk _ H1)
    .ok (BV16.ofBytes x0.toBV8 x1.toBV8)
  else .error s!"Index out of range: {n}"

abbrev BV32 := BitVec 32

def BV32.toByteArray (x : BV32) : ByteArray := bitVecToByteArray 32 x

def BV32.toBytes (n : BV32) : BV8 × BV8 × BV8 × BV8 :=
  let n0 := (n >>> 0o00 &&& 0xFF).truncate 8
  let n1 := (n >>> 0o10 &&& 0xFF).truncate 8
  let n2 := (n >>> 0o20 &&& 0xFF).truncate 8
  let n3 := (n >>> 0o30 &&& 0xFF).truncate 8
  (n0, n1, n2, n3)

def BV32.ofBytes (x0 x1 x2 x3 : BV8) : BV32 :=
  (x0.zeroExtend 32 <<< 0o00) |||
  (x1.zeroExtend 32 <<< 0o10) |||
  (x2.zeroExtend 32 <<< 0o20) |||
  (x3.zeroExtend 32 <<< 0o30)

theorem BV32.BytesRoundTrip (n : BV32) :
  let (x0, x1, x2, x3) := BV32.toBytes n
  let n' := BV32.ofBytes x0 x1 x2 x3
  n = n' := by
    unfold BV32.toBytes BV32.ofBytes
    bv_decide

theorem BV32.BytesRoundTrip1 (x0 x1 x2 x3 : BV8) :
  let n := BV32.ofBytes x0 x1 x2 x3
  let (x0', x1', x2', x3') := BV32.toBytes n
  x0 = x0' &&
  x1 = x1' &&
  x2 = x2' &&
  x3 = x3' := by
    unfold BV32.toBytes BV32.ofBytes
    bv_decide

def _root_.ByteArray.toBV32 (x : ByteArray) (startIndex : Nat) : Err BV32 :=
  let n := startIndex
  if H7 : n + 3 < x.size then
    let H0 : n + 0 < x.size := by omega
    let H1 : n + 1 < x.size := by omega
    let H2 : n + 2 < x.size := by omega
    let H3 : n + 3 < x.size := by omega
    let x0 := x.get (Fin.mk _ H0)
    let x1 := x.get (Fin.mk _ H1)
    let x2 := x.get (Fin.mk _ H2)
    let x3 := x.get (Fin.mk _ H3)
    .ok (BV32.ofBytes x0.toBV8 x1.toBV8 x2.toBV8 x3.toBV8)
else .error s!"Index out of range: {n}"

abbrev BV64 := BitVec 64

def BV64.toByteArray (x : BV64) : ByteArray := bitVecToByteArray 64 x

def BV64.ofNat (i : Nat) : BV64 := i.toUInt64.toBitVec

def BV64.ofInt (i : Int) : BV64 := i.toInt64.toBitVec

def BV64.toBytes (n : BV64) : BV8 × BV8 × BV8 × BV8 × BV8 × BV8 × BV8 × BV8 :=
  let n0 := (n >>> 0o00 &&& 0xFF).truncate 8
  let n1 := (n >>> 0o10 &&& 0xFF).truncate 8
  let n2 := (n >>> 0o20 &&& 0xFF).truncate 8
  let n3 := (n >>> 0o30 &&& 0xFF).truncate 8
  let n4 := (n >>> 0o40 &&& 0xFF).truncate 8
  let n5 := (n >>> 0o50 &&& 0xFF).truncate 8
  let n6 := (n >>> 0o60 &&& 0xFF).truncate 8
  let n7 := (n >>> 0o70 &&& 0xFF).truncate 8
  (n0, n1, n2, n3, n4, n5, n6, n7)

def BV64.ofBytes (x0 x1 x2 x3 x4 x5 x6 x7 : BV8) : BV64 :=
  (x0.zeroExtend 64 <<< 0o00) |||
  (x1.zeroExtend 64 <<< 0o10) |||
  (x2.zeroExtend 64 <<< 0o20) |||
  (x3.zeroExtend 64 <<< 0o30) |||
  (x4.zeroExtend 64 <<< 0o40) |||
  (x5.zeroExtend 64 <<< 0o50) |||
  (x6.zeroExtend 64 <<< 0o60) |||
  (x7.zeroExtend 64 <<< 0o70)

theorem BV64.BytesRoundTrip (n : BV64) :
  let (x0, x1, x2, x3, x4, x5, x6, x7) := BV64.toBytes n
  let n' := BV64.ofBytes x0 x1 x2 x3 x4 x5 x6 x7
  n = n' := by
    unfold BV64.toBytes BV64.ofBytes
    bv_decide

theorem BV64.BytesRoundTrip1 (x0 x1 x2 x3 x4 x5 x6 x7 : BV8) :
  let n := BV64.ofBytes x0 x1 x2 x3 x4 x5 x6 x7
  let (x0', x1', x2', x3', x4', x5', x6', x7') := BV64.toBytes n
  x0 = x0' &&
  x1 = x1' &&
  x2 = x2' &&
  x3 = x3' &&
  x4 = x4' &&
  x5 = x5' &&
  x6 = x6' &&
  x7 = x7' := by
    unfold BV64.toBytes BV64.ofBytes
    bv_decide

def _root_.ByteArray.toBV64 (x : ByteArray) (startIndex : Nat) : Err BV64 :=
  let n := startIndex
  if H7 : n + 7 < x.size then
    let H0 : n + 0 < x.size := by omega
    let H1 : n + 1 < x.size := by omega
    let H2 : n + 2 < x.size := by omega
    let H3 : n + 3 < x.size := by omega
    let H4 : n + 4 < x.size := by omega
    let H5 : n + 5 < x.size := by omega
    let H6 : n + 6 < x.size := by omega
    let x0 := x.get (Fin.mk _ H0)
    let x1 := x.get (Fin.mk _ H1)
    let x2 := x.get (Fin.mk _ H2)
    let x3 := x.get (Fin.mk _ H3)
    let x4 := x.get (Fin.mk _ H4)
    let x5 := x.get (Fin.mk _ H5)
    let x6 := x.get (Fin.mk _ H6)
    let x7 := x.get (Fin.mk _ H7)
    .ok (BV64.ofBytes x0.toBV8 x1.toBV8 x2.toBV8 x3.toBV8 x4.toBV8 x5.toBV8 x6.toBV8 x7.toBV8)
  else .error s!"Index out of range: {n}"

/-
The largest Nat such that it and every smaller Nat can be represented exactly in a 64-bit IEEE-754 float
https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number/MAX_SAFE_INTEGER

The number is one less than 2^(mantissa size)
https://en.wikipedia.org/wiki/Floating-point_arithmetic
https://en.wikipedia.org/wiki/Double-precision_floating-point_format
-/
private def floatMantissaBits : Nat := 52
private def float32MantissaBits : Nat := 23
-- Add 1 to the mantissa length because of the implicit leading 1
def maxSafeNatForFloat : Nat := Nat.pow 2 (floatMantissaBits + 1) - 1
def maxSafeNatForFloat32 : Nat := Nat.pow 2 (float32MantissaBits + 1) - 1

def _root_.Int.toFloat (n : Int) : Float := Float.ofInt n
-- TODO: Use Flaot32.ofInt when https://github.com/leanprover/lean4/pull/7277 is merged, probably in 4.17.0
def _root_.Int.toFloat32 (n : Int) : Float32 := match n with
| Int.ofNat n => Float32.ofNat n
| Int.negSucc n => Float32.neg (Float32.ofNat (Nat.succ n))


def _root_.Float32.toLEByteArray (f : Float32) : ByteArray := bitVecToByteArray 32 f.toBits.toBitVec
def _root_.Float.toLEByteArray (f : Float) : ByteArray := bitVecToByteArray 64 f.toBits.toBitVec

/-- Interpret a `ByteArray` of size 4 as a little-endian `UInt32`. Missing from Lean stdlib. -/
def _root_.ByteArray.toUInt32LE! (bs : ByteArray) : UInt32 :=
  assert! bs.size == 4
  (bs.get! 3).toUInt32 <<< 0x18 |||
  (bs.get! 2).toUInt32 <<< 0x10 |||
  (bs.get! 1).toUInt32 <<< 0x8  |||
  (bs.get! 0).toUInt32

/-- Interpret a `ByteArray` of size 4 as a big-endian `UInt32`.  Missing from Lean stdlib. -/
def _root_.ByteArray.toUInt32BE! (bs : ByteArray) : UInt32 :=
  assert! bs.size == 4
  (bs.get! 0).toUInt32 <<< 0x38 |||
  (bs.get! 1).toUInt32 <<< 0x30 |||
  (bs.get! 2).toUInt32 <<< 0x28 |||
  (bs.get! 3).toUInt32 <<< 0x20

-- Assumes arr.size == 4
def Float32.ofLEByteArray! (arr : ByteArray) : Float32 :=
  Float32.ofBits arr.toUInt32LE!

-- Assumes arr.size == 8
def Float.ofLEByteArray! (arr : ByteArray) : Float :=
  Float.ofBits arr.toUInt64LE!

def _root_.Float32.toNat (f : Float32) : Nat := f.toUInt64.toNat

def _root_.Float32.toInt (f : Float32) : Int :=
  let neg := f <= 0
  let f := if neg then -f else f
  let n := Int.ofNat f.toUInt64.toNat
  if neg then -n else n

def _root_.Float.toNat (f : Float) : Nat := f.toUInt32.toNat

def _root_.Float.toInt (f : Float) : Int :=
  let neg := f <= 0
  let f := if neg then -f else f
  let n := Int.ofNat f.toUInt64.toNat
  if neg then -n else n

#guard (
  let n : BV64 := 0x3FFAB851EB851EB8
  do
    let arr := n.toByteArray
    let n' <- ByteArray.toBV64 arr 0
    return n == n') == .ok true

end TensorLib
