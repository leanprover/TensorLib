/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/
import Plausible
import TensorLib.Common
import TensorLib.Shape

open Plausible(Gen SampleableExt Shrinkable)

namespace TensorLib

/-! The subset of types NumPy supports that we care about -/
inductive Dtype where
| bool
| int8
| int16
| int32
| int64
| uint8
| uint16
| uint32
| uint64
| float32
| float64
deriving BEq, Repr, Inhabited


namespace Dtype

instance : LawfulBEq Dtype where
  eq_of_beq {a b} h := by cases a <;> cases b <;> first | rfl | contradiction
  rfl {a} := by cases a <;> decide

def gen : Gen Dtype := Gen.elements [
  bool,
  int8,
  int16,
  int32,
  int64,
  uint8,
  uint16,
  uint32,
  uint64,
  float32,
  float64
] (by simp)

instance : Shrinkable Dtype where

instance : SampleableExt Dtype := SampleableExt.mkSelfContained gen

-- Should match the NumPy name of the dtype. We use toString to generate NumPy test code.
instance : ToString Dtype where
  toString
  | .bool => "bool"
  | int8 => "int8"
  | int16 => "int16"
  | int32 => "int32"
  | int64 => "int64"
  | uint8 => "uint8"
  | uint16 => "uint16"
  | uint32 => "uint32"
  | uint64 => "uint64"
  | float32 => "float32"
  | float64 => "float64"

def isOneByte (x : Dtype) : Bool := match x with
| bool | int8 | uint8 => true
| _ => false

def isMultiByte (x : Dtype) : Bool := ! x.isOneByte

def isInt (x : Dtype) : Bool := match x with
| int8 | int16 | int32 | int64 => true
| _ => false

def isUint (x : Dtype) : Bool := match x with
| uint8 | uint16 | uint32 | uint64 => true
| _ => false

def isIntLike (x : Dtype) : Bool := x.isInt || x.isUint

--! Number of bytes used by each element of the given dtype
def itemsize (x : Dtype) : Nat := match x with
| float64 | int64 | uint64 => 8
| float32 | int32 | uint32 => 4
| int16 | uint16 => 2
| bool | int8 | uint8 => 1

def lossless (fromDtype toDtype : Dtype) : Bool := match fromDtype, toDtype with
| .bool, _ => true
| _, .bool => false
| .int8, .int8
| .int8, .int16
| .int8, .int32
| .int8, .int64
| .int8, .float32
| .int8, .float64 => true
| .int8, _ => false
| .uint8, .uint8
| .uint8, .uint16
| .uint8, .uint32
| .uint8, .uint64
| .uint8, .int16
| .uint8, .int32
| .uint8, .int64
| .uint8, .float32
| .uint8, .float64 => true
| .uint8, _ => false
| .int16, .int16
| .int16, .int32
| .int16, .int64
| .int16, .float32
| .int16, .float64 => true
| .int16, _ => false
| .uint16, .uint16
| .uint16, .uint32
| .uint16, .uint64
| .uint16, .int32
| .uint16, .int64
| .uint16, .float32
| .uint16, .float64 => true
| .uint16, _ => false
| .int32, .int32
| .int32, .int64
| .int32, .float64 => true
| .int32, _ => false
| .uint32, .uint32
| .uint32, .uint64
| .uint32, .float64 => true
| .uint32, _ => false
| .int64, .int64 => true
| .int64, _ => false
| .uint64, .uint64 => true
| .uint64, _ => false
| .float32, .float32
| .float32, .float64 => true
| .float32, _ => false
| .float64, .float64 => true
| .float64, _ => false

theorem losslessAntiSymmetric (t1 t2 : Dtype) (H : lossless t1 t2) (H1 : t1 ≠ t2) : !(lossless t2 t1) := by
  revert H
  unfold lossless;
  cases t1 <;> cases t2 <;> trivial

/-
In NumPy, I could not get bool coversion to crash. Any non-zero value becomes True, 0 becomes false

# np.array(0x100, dtype='bool')
array(True)

# np.array(0xFF, dtype='uint8')
array(255, dtype=uint8)

# np.array(0x100, dtype='uint8')
OverflowError: Python integer 256 out of bounds for uint8

# np.array(0x7F, dtype='int8')
array(127, dtype=int8)

# np.array(0x80, dtype='int8')
OverflowError: Python integer 128 out of bounds for int8

... same patterns for {u,}int{16,32,64}

Float types have named safe nat upper bounds.
-/
private def maxSafeNat : Dtype -> Option Nat
| .bool => none
| .uint8 => some 0xFF
| .int8 => some 0x7F
| .uint16 => some 0xFFFF
| .int16 => some 0x7FFF
| .uint32 => some 0xFFFFFFFF
| .int32 => some 0x7FFFFFFF
| .uint64 => some 0xFFFFFFFFFFFFFFFF
| .int64 => some 0x7FFFFFFFFFFFFFFF
| .float32 => maxSafeNatForFloat32
| .float64 => maxSafeNatForFloat64

private def canCastFromNat (dtype : Dtype) (n : Nat) : Bool := n <= dtype.maxSafeNat.getD n

/-
NumPy doesn't allow casts from negative numbers to uint types, even if they fit.
# np.array(-0x1, dtype='uint8')
OverflowError: Python integer -1 out of bounds for uint8
-/
private def minSafeInt : Dtype -> Option Int
| .bool => none
| .uint8 | .uint16 | .uint32 | .uint64 => some 0
| .int8 => some (-0x80)
| .int16 => some (-0x8000)
| .int32 => some (-0x80000000)
| .int64 => some (-0x8000000000000000)
| .float32 => some (-maxSafeNatForFloat32)
| .float64 => some (-maxSafeNatForFloat64)

private def canCastFromInt (dtype : Dtype) (n : Int) : Bool :=
  if n < 0 then dtype.minSafeInt.getD n <= n
  else n <= dtype.maxSafeNat.getD n.toNat

def sizedStrides (dtype : Dtype) (s : Shape) : Strides := List.map (fun x => x * dtype.itemsize) s.unitStrides

private def byteArrayOfNatOverflow (dtype : Dtype) (n : Nat) : ByteArray := match dtype with
| .bool => (BV8.ofNat $ if n == 0 then 0 else 1).toByteArray
| .uint8 => (BV8.ofNat n).toByteArray
| .int8 => [(Int8.ofNat n).toUInt8].toByteArray
| .uint16 => BV16.toByteArray n.toUInt16.toBitVec
| .int16 => BV16.toByteArray n.toInt16.toBitVec
| .uint32 => BV32.toByteArray n.toUInt32.toBitVec
| .int32 => BV32.toByteArray n.toInt32.toBitVec
| .uint64 => BV64.toByteArray n.toUInt64.toBitVec
| .int64 => BV64.toByteArray n.toInt64.toBitVec
| .float32 => n.toFloat32.toLEByteArray
| .float64 => n.toFloat.toLEByteArray

def byteArrayOfNat (dtype : Dtype) (n : Nat) : Err ByteArray :=
  if dtype.canCastFromNat n then .ok (dtype.byteArrayOfNatOverflow n)
  else .error s!"Nat {n} out of bounds for {dtype}"

def byteArrayOfNat! (dtype : Dtype) (n : Nat) : ByteArray := get! $ byteArrayOfNat dtype n

private def byteArrayToNatRoundTrip (dtype : Dtype) (n : Nat) : Bool :=
  let res := do
    let arr <- dtype.byteArrayOfNat n
    let n' := arr.toNat
    return n == n'
  res.toOption.getD false

#guard uint8.byteArrayToNatRoundTrip 0
#guard uint8.byteArrayToNatRoundTrip 5
#guard uint8.byteArrayToNatRoundTrip 255
#guard !uint8.byteArrayToNatRoundTrip 256

private def byteArrayOfIntOverflow (dtype : Dtype) (n : Int) : ByteArray := match dtype with
| .bool => (BV8.ofNat $ if n == 0 then 0 else 1).toByteArray
| .uint8 | .int8 => [n.toInt8.toUInt8].toByteArray
| .uint16 | .int16 => BV16.toByteArray n.toInt16.toBitVec
| .uint32 | .int32 => BV32.toByteArray n.toInt32.toBitVec
| .uint64 | .int64 => BV64.toByteArray n.toInt64.toBitVec
| .float32 => n.toFloat32.toLEByteArray
| .float64 => n.toFloat64.toLEByteArray

def byteArrayOfInt (dtype : Dtype) (n : Int) : Err ByteArray :=
  if dtype.canCastFromInt n then .ok (dtype.byteArrayOfIntOverflow n)
  else .error s!"Int {n} out of bounds for {dtype}"

def byteArrayOfInt! (dtype : Dtype) (n : Int) : ByteArray := get! $ byteArrayOfInt dtype n

private def byteArrayToIntRoundTrip (dtype : Dtype) (n : Int) : Bool :=
  let res := do
    let arr <- dtype.byteArrayOfInt n
    let n' := arr.toInt
    return n == n'
  res.toOption.getD false

#guard int8.byteArrayToIntRoundTrip 0
#guard int8.byteArrayToIntRoundTrip 5
#guard int8.byteArrayToIntRoundTrip 127
#guard !int8.byteArrayToIntRoundTrip 255

private def byteArrayToFloat (dtype : Dtype) (arr : ByteArray) : Err Float := match dtype with
| .float64 =>
  if arr.size != 8 then .error "byte size mismatch" else
  .ok $ Float.ofBits arr.toUInt64LE! -- toUInt64LE! is ok here because we already checked the size
| _ => .error "Illegal type conversion"

private def byteArrayToFloat! (dtype : Dtype) (arr : ByteArray) : Float := get! $ byteArrayToFloat dtype arr

def byteArrayOfFloat (dtype : Dtype) (f : Float) : Err ByteArray := match dtype with
| .float64 => .ok $ BV64.toByteArray f.toBits.toBitVec
| _ => .error "Illegal type conversion"

private def byteArrayOfFloat! (dtype : Dtype) (f : Float) : ByteArray := get! $ byteArrayOfFloat dtype f

def byteArrayOfFloat32 (dtype : Dtype) (f : Float32) : Err ByteArray := match dtype with
| .float32 => .ok $ BV32.toByteArray f.toBits.toBitVec
| _ => .error "Illegal type conversion"

private def byteArrayOfFloat32! (dtype : Dtype) (f : Float32) : ByteArray := get! $ byteArrayOfFloat32 dtype f

private def byteArrayToFloatRoundTrip (dtype : Dtype) (f : Float) : Bool :=
  let res := do
    let arr <- dtype.byteArrayOfFloat f
    let f' <- dtype.byteArrayToFloat arr
    return f == f'
  res.toOption.getD false

#guard float64.byteArrayToFloatRoundTrip 0
#guard float64.byteArrayToFloatRoundTrip 0.1
#guard float64.byteArrayToFloatRoundTrip (-0)
#guard float64.byteArrayToFloatRoundTrip 17
#guard float64.byteArrayToFloatRoundTrip (Float.sqrt 2)
#guard !float32.byteArrayToFloatRoundTrip 0

def byteArrayToFloat32 (dtype : Dtype) (arr : ByteArray) : Err Float32 := match dtype with
| .float32 => arr.toUInt32LE.map Float32.ofBits
| _ => .error "Illegal type conversion"

def byteArrayToFloat32! (dtype : Dtype) (arr : ByteArray) : Float32 :=  get! $ byteArrayToFloat32 dtype arr

private def byteArrayToFloat32RoundTrip (dtype : Dtype) (f : Float32) : Bool :=
  let res := do
    let arr <- dtype.byteArrayOfFloat32 f
    let f' <- dtype.byteArrayToFloat32 arr
    return f == f'
  res.toOption.getD false

#guard !float64.byteArrayToFloat32RoundTrip 0
#guard float32.byteArrayToFloat32RoundTrip 0.1
#guard float32.byteArrayToFloat32RoundTrip (-0)
#guard float32.byteArrayToFloat32RoundTrip 17
#guard float32.byteArrayToFloat32RoundTrip (Float32.sqrt 2)
#guard float32.byteArrayToFloat32RoundTrip 0

/-
NumPy addition overflows and underflows without complaint. We will do the same.
-/
def add (dtype : Dtype) (x y : ByteArray) : Err ByteArray :=
  if dtype.itemsize != x.size || dtype.itemsize != y.size then .error s!"add: byte size mismatch: {dtype} {x.size} {y.size}" else
  match dtype with
  | .bool => do
    let x := x.toNat
    let y := y.toNat
    if x == 1 || y == 1 then dtype.byteArrayOfInt 1
    else if x == 0 && y == 0 then dtype.byteArrayOfInt 0
    else .error "illegal bool bytes"
  | .uint8 | .uint16 | .uint32 | .uint64 => do
    return dtype.byteArrayOfNatOverflow (x.toNat + y.toNat)
  | .int8 | .int16| .int32 | .int64 => do
    dtype.byteArrayOfInt (x.toInt + y.toInt)
  | .float32 => do
    let x <- dtype.byteArrayToFloat32 x
    let y <- dtype.byteArrayToFloat32 y
    dtype.byteArrayOfFloat32 (x + y)
  | .float64 => do
    let x <- dtype.byteArrayToFloat x
    let y <- dtype.byteArrayToFloat y
    dtype.byteArrayOfFloat (x + y)

def sub (dtype : Dtype) (x y : ByteArray) : Err ByteArray :=
  if dtype.itemsize != x.size || dtype.itemsize != y.size then .error "sub: byte size mismatch" else
  match dtype with
  | .uint8 | .uint16 | .uint32 | .uint64 => do
    return dtype.byteArrayOfNatOverflow (x.toNat - y.toNat)
  | .int8 | .int16| .int32 | .int64 => do
    return dtype.byteArrayOfIntOverflow (x.toInt - y.toInt)
  | .float32 => do
    let x <- dtype.byteArrayToFloat32 x
    let y <- dtype.byteArrayToFloat32 y
    dtype.byteArrayOfFloat32 (x - y)
  | .float64 => do
    let x <- dtype.byteArrayToFloat x
    let y <- dtype.byteArrayToFloat y
    dtype.byteArrayOfFloat (x - y)
  | .bool => .error s!"`sub` not supported at type ${dtype}"

def mul (dtype : Dtype) (x y : ByteArray) : Err ByteArray :=
  if dtype.itemsize != x.size || dtype.itemsize != y.size then .error "mul: byte size mismatch" else
  match dtype with
  | .uint8 | .uint16 | .uint32 | .uint64 => do
    return dtype.byteArrayOfNatOverflow (x.toNat * y.toNat)
  | .int8 | .int16| .int32 | .int64 => do
    return dtype.byteArrayOfIntOverflow (x.toInt * y.toInt)
  | .float32 => do
    let x <- dtype.byteArrayToFloat32 x
    let y <- dtype.byteArrayToFloat32 y
    dtype.byteArrayOfFloat32 (x * y)
  | .float64 => do
    let x <- dtype.byteArrayToFloat x
    let y <- dtype.byteArrayToFloat y
    dtype.byteArrayOfFloat (x * y)
  | .bool => .error s!"`mul` not supported at type ${dtype}"

def div (dtype : Dtype) (x y : ByteArray) : Err ByteArray :=
  if dtype.itemsize != x.size || dtype.itemsize != y.size then .error "div: byte size mismatch" else
  match dtype with
  | .uint8 | .uint16 | .uint32 | .uint64 => do
    return dtype.byteArrayOfNatOverflow (x.toNat / y.toNat)
  | .int8 | .int16| .int32 | .int64 => do
    return dtype.byteArrayOfIntOverflow (x.toInt / y.toInt)
  | .float32 => do
    let x <- dtype.byteArrayToFloat32 x
    let y <- dtype.byteArrayToFloat32 y
    dtype.byteArrayOfFloat32 (x / y)
  | .float64 => do
    let x <- dtype.byteArrayToFloat x
    let y <- dtype.byteArrayToFloat y
    dtype.byteArrayOfFloat (x / y)
  | .bool => .error s!"`div` not supported at type ${dtype}"

/-
This works for int/uint/bool/float. Keep an eye out when we start implementing unusual floating point types.
-/
def zero (dtype : Dtype) : ByteArray := ByteArray.mk $ (List.replicate dtype.itemsize (0 : UInt8)).toArray

def abs (dtype : Dtype) (x : ByteArray) : Err ByteArray := do
  match dtype with
  | .uint8 | .uint16 | .uint32 | .uint64 => return x
  | .int8 | .int16| .int32 | .int64 => return dtype.byteArrayOfIntOverflow x.toInt.natAbs
  | .float32 =>
    let x <- dtype.byteArrayToFloat32 x
    dtype.byteArrayOfFloat32 x.abs
  | .float64 => do
    let x <- dtype.byteArrayToFloat x
    dtype.byteArrayOfFloat x.abs
  | .bool => return x

def castOverflow (fromDtype : Dtype) (data : ByteArray) (toDtype : Dtype) : Err ByteArray :=
  if fromDtype == toDtype then return data else
  match fromDtype, toDtype with
  | _, .bool => return ByteArray.mk #[if data.data.all fun x => x == 0 then 0 else 1]
  | .bool, _ | .uint8, _ | .uint16, _ | .uint32, _ | .uint64, _ =>
    return toDtype.byteArrayOfNatOverflow data.toNat
  | .int8, _ | .int16, _ | .int32, _ | .int64, _ =>
    return toDtype.byteArrayOfIntOverflow data.toInt
  | .float32, .uint8 | .float32, .uint16  | .float32, .uint32 | .float32, .uint64 => do
    let f <- Float32.ofLEByteArray data
    return toDtype.byteArrayOfNatOverflow f.toNat
  | .float32, .int8 | .float32, .int16  | .float32, .int32 | .float32, .int64 => do
    let f <- Float32.ofLEByteArray data
    return toDtype.byteArrayOfIntOverflow f.toInt
  | .float64, .uint8 | .float64, .uint16  | .float64, .uint32 | .float64, .uint64 => do
    let f <- Float.ofLEByteArray data
    return toDtype.byteArrayOfNatOverflow f.toNat
  | .float64, .int8 | .float64, .int16  | .float64, .int32 | .float64, .int64 => do
    let f <- Float.ofLEByteArray data
    return toDtype.byteArrayOfIntOverflow f.toInt
  | .float32, .float64 => do
    let f <- Float32.ofLEByteArray data
    return f.toFloat.toLEByteArray
  | .float64, .float32 => do
    let f <- Float.ofLEByteArray data
    return f.toFloat32.toLEByteArray
  | .float32, .float32 | .float64, .float64 => impossible

section Test

open Plausible

private def canCastLosslessRoundTrip (fromDtype : Dtype) (data : ByteArray) (toDtype : Dtype) : Bool :=
  let res := do
    let x <- castOverflow fromDtype data toDtype
    let y <- castOverflow toDtype x fromDtype
    return data == y
  match res with
  | .ok b => b
  | .error _ => false

private def canCastLosslessIntRoundTrip (fromDtype : Dtype) (n : Int) (toDtype : Dtype) : Bool :=
  let res := do
    let n <- fromDtype.byteArrayOfInt n
    return canCastLosslessRoundTrip fromDtype n toDtype
  match res with
  | .ok b => b
  | .error _ => false

#guard
  let fromDtype := Dtype.int8
  let toDtype := Dtype.float32
  let n : Int := 5
  canCastLosslessIntRoundTrip fromDtype n toDtype

#guard
  let fromDtype := Dtype.int8
  let toDtype := Dtype.float32
  let n : Int := -5
  canCastLosslessIntRoundTrip fromDtype n toDtype

#guard
  let fromDtype := Dtype.int8
  let toDtype := Dtype.float32
  let n : Int := 0x7F
  canCastLosslessIntRoundTrip fromDtype n toDtype

#guard
  let fromDtype := Dtype.int8
  let toDtype := Dtype.float32
  let n : Int := 0x80
  !canCastLosslessIntRoundTrip fromDtype n toDtype

#guard
  let fromDtype := Dtype.uint8
  let toDtype := Dtype.float32
  let n : Int := 0xFF
  canCastLosslessIntRoundTrip fromDtype n toDtype

#guard
  let fromDtype := Dtype.uint16
  let toDtype := Dtype.float32
  let n : Int := 0xFFFF
  canCastLosslessIntRoundTrip fromDtype n toDtype

#guard
  let fromDtype := Dtype.uint32
  let toDtype := Dtype.float32
  let n : Int := maxSafeNatForFloat32
  canCastLosslessIntRoundTrip fromDtype n toDtype

#guard
  let fromDtype := Dtype.uint32
  let toDtype := Dtype.float64
  let n : Int := min maxSafeNatForFloat64 0xFFFFFFFF
  canCastLosslessIntRoundTrip fromDtype n toDtype

#guard
  let fromDtype := Dtype.uint64
  let toDtype := Dtype.float64
  let n : Int := maxSafeNatForFloat64
  canCastLosslessIntRoundTrip fromDtype n toDtype

-- 0 and 1 should be translatable at any dtype
/--
warning: declaration uses 'sorry'
-/
#guard_msgs in
example (fromDtype toDtype : Dtype) (n : Nat) :
  canCastLosslessIntRoundTrip fromDtype 0 toDtype &&
  canCastLosslessIntRoundTrip fromDtype 1 toDtype
  := by
  plausible (config := cfg)

-- One dtype should always go back and forth
/--
warning: declaration uses 'sorry'
-/
#guard_msgs in
example (dtype : Dtype) (n : Nat) :
  canCastLosslessIntRoundTrip dtype n dtype := by
  plausible (config := cfg)

-- Lossless translations should be OK
/--
warning: declaration uses 'sorry'
-/
#guard_msgs in
example (fromDtype toDtype : Dtype) (n : Nat) :
  if lossless fromDtype toDtype then
    canCastLosslessIntRoundTrip fromDtype n toDtype
  else true := by
  plausible (config := cfg)

end Test

end Dtype
end TensorLib
