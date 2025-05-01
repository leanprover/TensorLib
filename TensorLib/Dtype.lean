/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/
import Plausible
import TensorLib.ByteArray
import TensorLib.Bytes
import TensorLib.Common
import TensorLib.Float
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

def isFloat (x : Dtype) : Bool := match x with
| .float32 | .float64 => true
| _ => false

--! Number of bytes used by each element of the given dtype
def itemsize (x : Dtype) : Nat := match x with
| float64 | int64 | uint64 => 8
| float32 | int32 | uint32 => 4
| int16 | uint16 => 2
| bool | int8 | uint8 => 1

-- This is the type NumPy returns when using binary operators on arrays
-- with the given types. E.g. uint16 and int16 returns an int32.
def join (x y : Dtype) : Option Dtype :=
  let (x, y) := if x.itemsize <= y.itemsize then (x, y) else (y, x)
  if x == y then x else
  match x, y with
  | float32, float64 => float64
  | float32, _
  | _, float32
  | float64, _
  | _, float64 => none
  | bool, _ => y
  | int8, uint8
  | uint8, int8 => int16
  | int8, _
  | uint8, _ => y
  | int16, uint16
  | uint16, int16 => int32
  | int16, _
  | uint16, _ => y
  | int32, uint32
  | uint32, int32 => int64
  | int32, _
  | uint32, _ => y
  | int64, _
  | _, int64
  | uint64, _ => none -- NumPy gives "can't safely coerce" for uint64/int64

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

def byteArrayOfNatOverflow (dtype : Dtype) (n : Nat) : ByteArray := match dtype with
| .bool => (if n == 0 then 0 else 1).toUInt8.toLEByteArray
| .uint8 => n.toUInt8.toLEByteArray
| .int8 => n.toInt8.toLEByteArray
| .uint16 => n.toUInt16.toLEByteArray
| .int16 => n.toInt16.toLEByteArray
| .uint32 => n.toUInt32.toLEByteArray
| .int32 => n.toInt32.toLEByteArray
| .uint64 => n.toUInt64.toLEByteArray
| .int64 => n.toInt64.toLEByteArray
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

private def byteArrayToFloat64 (dtype : Dtype) (arr : ByteArray) : Err Float := match dtype with
| .float64 =>
  if arr.size != 8 then .error "byte size mismatch" else
  .ok $ Float.ofBits arr.toUInt64LE! -- toUInt64LE! is ok here because we already checked the size
| _ => .error "Illegal type conversion"

private def byteArrayToFloat64! (dtype : Dtype) (arr : ByteArray) : Float := get! $ byteArrayToFloat64 dtype arr

def byteArrayOfFloat64 (dtype : Dtype) (f : Float) : Err ByteArray := match dtype with
| .float64 => .ok $ BV64.toByteArray f.toBits.toBitVec
| _ => .error "Illegal type conversion"

private def byteArrayOfFloat64! (dtype : Dtype) (f : Float) : ByteArray := get! $ byteArrayOfFloat64 dtype f

def byteArrayOfFloat32 (dtype : Dtype) (f : Float32) : Err ByteArray := match dtype with
| .float32 => .ok $ BV32.toByteArray f.toBits.toBitVec
| _ => .error "Illegal type conversion"

private def byteArrayOfFloat32! (dtype : Dtype) (f : Float32) : ByteArray := get! $ byteArrayOfFloat32 dtype f

private def byteArrayToFloat64RoundTrip (dtype : Dtype) (f : Float) : Bool :=
  let res := do
    let arr <- dtype.byteArrayOfFloat64 f
    let f' <- dtype.byteArrayToFloat64 arr
    return f == f'
  res.toOption.getD false

#guard float64.byteArrayToFloat64RoundTrip 0
#guard float64.byteArrayToFloat64RoundTrip 0.1
#guard float64.byteArrayToFloat64RoundTrip (-0)
#guard float64.byteArrayToFloat64RoundTrip 17
#guard float64.byteArrayToFloat64RoundTrip (Float.sqrt 2)
#guard !float32.byteArrayToFloat64RoundTrip 0

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
    let x <- dtype.byteArrayToFloat64 x
    let y <- dtype.byteArrayToFloat64 y
    dtype.byteArrayOfFloat64 (x + y)

def add! (dtype : Dtype) (x y : ByteArray) : ByteArray := get! $ add dtype x y

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
    let x <- dtype.byteArrayToFloat64 x
    let y <- dtype.byteArrayToFloat64 y
    dtype.byteArrayOfFloat64 (x - y)
  | .bool => .error s!"`sub` not supported at type ${dtype}"

def sub! (dtype : Dtype) (x y : ByteArray) : ByteArray := get! $ sub dtype x y

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
    let x <- dtype.byteArrayToFloat64 x
    let y <- dtype.byteArrayToFloat64 y
    dtype.byteArrayOfFloat64 (x * y)
  | .bool => .error s!"`mul` not supported at type ${dtype}"

def mul! (dtype : Dtype) (x y : ByteArray) : ByteArray := get! $ mul dtype x y

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
    let x <- dtype.byteArrayToFloat64 x
    let y <- dtype.byteArrayToFloat64 y
    dtype.byteArrayOfFloat64 (x / y)
  | .bool => .error s!"`div` not supported at type ${dtype}"

def div! (dtype : Dtype) (x y : ByteArray) : ByteArray := get! $ div dtype x y

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
    let x <- dtype.byteArrayToFloat64 x
    dtype.byteArrayOfFloat64 x.abs
  | .bool => return x

def abs! (dtype : Dtype) (x : ByteArray) : ByteArray := get! $ abs dtype x

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

def isZero (dtype : Dtype) (x : ByteArray) : Err Bool := match dtype with
| bool
| int8
| uint8
| int16
| uint16
| int32
| uint32
| int64
| uint64 => return x.data.all fun v => v == 0
-- We need to worry about -0, which is not all 0s in the bit pattern.
| float32 => do
  let f <- Float32.ofLEByteArray x
  return f == 0
| float64 => do
  let f <- Float.ofLEByteArray x
  return f == 0

def isZero! (dtype : Dtype) (x : ByteArray) : Bool := get! $ dtype.isZero x

def nonZero (dtype : Dtype) (x : ByteArray) : Err Bool := (dtype.isZero x).map not

def nonZero! (dtype : Dtype) (x : ByteArray) : Bool := get! $ dtype.nonZero x

-- In one sense it's annoying to have an abbreviation, but it also saves some
-- cognitive effort required to translate between numbers and bools.
def logicalNot : Dtype -> ByteArray -> Err Bool := isZero

#guard Dtype.uint64.isZero! (0 : UInt64).toLEByteArray
#guard Dtype.uint64.isZero! (-0 : Int64).toLEByteArray
#guard (0.0 : Float32).toLEByteArray.data.all fun x => x == 0
#guard !(-0.0 : Float32).toLEByteArray.data.all fun x => x == 0
#guard (0.0 : Float).toLEByteArray.data.all fun x => x == 0
#guard !(-0.0 : Float).toLEByteArray.data.all fun x => x == 0
#guard Dtype.float32.isZero! (-0.0 : Float32).toLEByteArray
#guard Dtype.float64.isZero! (-0.0 : Float).toLEByteArray

private def logicalBinop (f : Bool -> Bool -> Bool) (t1 : Dtype) (x1 : ByteArray) (t2 : Dtype) (x2 : ByteArray) : Err Bool := do
  let z1 <- t1.nonZero x1
  let z2 <- t2.nonZero x2
  return f z1 z2

def logicalAnd : Dtype -> ByteArray -> Dtype -> ByteArray -> Err Bool :=
  logicalBinop (fun x y => x && y)

def logicalAnd! (t1 : Dtype) (x1 : ByteArray) (t2 : Dtype) (x2 : ByteArray) : Bool :=
  get! $ logicalAnd t1 x1 t2 x2

def logicalOr : Dtype -> ByteArray -> Dtype -> ByteArray -> Err Bool :=
  logicalBinop (fun x y => x || y)

def logicalOr! (t1 : Dtype) (x1 : ByteArray) (t2 : Dtype) (x2 : ByteArray) : Bool :=
  get! $ logicalOr t1 x1 t2 x2

def logicalXor : Dtype -> ByteArray -> Dtype -> ByteArray -> Err Bool :=
  logicalBinop (fun x y => xor x y)

def logicalXor! (t1 : Dtype) (x1 : ByteArray) (t2 : Dtype) (x2 : ByteArray) : Bool :=
  get! $ logicalXor t1 x1 t2 x2

#guard logicalAnd! Dtype.uint8 (1:UInt8).toLEByteArray Dtype.float64 (5:Float).toLEByteArray
#guard !logicalAnd! Dtype.uint8 (1:UInt8).toLEByteArray Dtype.float64 (-0:Float).toLEByteArray
#guard logicalOr! Dtype.uint8 (1:UInt8).toLEByteArray Dtype.float64 (-0:Float).toLEByteArray
#guard logicalOr! Dtype.uint8 (0:UInt8).toLEByteArray Dtype.float64 (-0.1:Float).toLEByteArray
#guard !logicalOr! Dtype.uint8 (0:UInt8).toLEByteArray Dtype.float64 (-0.0:Float).toLEByteArray
#guard !logicalXor! Dtype.uint8 (0:UInt8).toLEByteArray Dtype.float64 (-0.0:Float).toLEByteArray

/-
When you call real functions on int arrays, for example, NumPy converts the array to some float
type before calling the function. This follows some rules, which we approximate here, given we don't
have all the types available in NumPy (e.g. float16) and we may have types like bfloat that aren't
in NumPy out of the box.
-/
def floatVariant (dtype : Dtype) : Dtype :=
  if dtype.itemsize <= 1 then Dtype.float32 -- leaving this branch in because should be Dtype.float16. TODO to change
  else if dtype.itemsize <= 2 then Dtype.float32
  else Dtype.float64

private def liftFloatUnop (f32 : Float32 -> Err Float32) (f64 : Float -> Err Float)
                          (dtype : Dtype) (data : ByteArray) : Err ByteArray := do
  if data.size != dtype.itemsize then throw "incorrect byte count" else
  match dtype with
  | .float32 => do
    let f <- Float32.ofLEByteArray data
    let x <- f32 f
    return x.toLEByteArray
  | .float64 => do
    let f <- Float.ofLEByteArray data
    let x <- f64 f
    return x.toLEByteArray
  | _ => throw "operation requires a float type"

def arctan : Dtype -> ByteArray -> Err ByteArray :=
  liftFloatUnop (comp .ok Float32.atan) (comp .ok Float.atan)

def arctan! (dtype : Dtype) (data : ByteArray) : ByteArray := get! $ arctan dtype data

def cos : Dtype -> ByteArray -> Err ByteArray :=
  liftFloatUnop (comp .ok Float32.cos) (comp .ok Float.cos)

def cos! (dtype : Dtype) (data : ByteArray) : ByteArray := get! $ cos dtype data

def exp : Dtype -> ByteArray -> Err ByteArray :=
  liftFloatUnop (comp .ok Float32.exp) (comp .ok Float.exp)

def exp! (dtype : Dtype) (data : ByteArray) : ByteArray := get! $ exp dtype data

def log : Dtype -> ByteArray -> Err ByteArray :=
  liftFloatUnop (comp .ok Float32.log) (comp .ok Float.log)

def log! (dtype : Dtype) (data : ByteArray) : ByteArray := get! $ log dtype data

def sin : Dtype -> ByteArray -> Err ByteArray :=
  liftFloatUnop (comp .ok Float32.sin) (comp .ok Float.sin)

def sin! (dtype : Dtype) (data : ByteArray) : ByteArray := get! $ sin dtype data

def sqrt : Dtype -> ByteArray -> Err ByteArray :=
  liftFloatUnop (comp .ok Float32.sqrt) (comp .ok Float.sqrt)

def sqrt! (dtype : Dtype) (data : ByteArray) : ByteArray := get! $ sqrt dtype data

def tan : Dtype -> ByteArray -> Err ByteArray :=
  liftFloatUnop (comp .ok Float32.tan) (comp .ok Float.tan)

def tan! (dtype : Dtype) (data : ByteArray) : ByteArray := get! $ tan dtype data

def tanh : Dtype -> ByteArray -> Err ByteArray :=
  liftFloatUnop (comp .ok Float32.tanh) (comp .ok Float.tanh)

def tanh! (dtype : Dtype) (data : ByteArray) : ByteArray := get! $ tanh dtype data

private def shift (f : UInt64 -> UInt64 -> UInt64) (dtype : Dtype) (bits : ByteArray) (shiftAmount : ByteArray) : Err ByteArray := match dtype with
| .float32 | .float64 => throw "shifts not supported at float type"
| .bool => throw "In NumPy, bool shifts are cast to int64. This seems arbitrary so please cast (e.g. with astype) before you shift."
| .uint64 | .int64 | .uint32 | .int32 | .uint16 | .int16 | .uint8 | .int8 =>
  let k := dtype.itemsize
  if bits.size != k then throw "dtype size mismatch" else
  let n64 := bits.toNat.toUInt64
  let shift64 := shiftAmount.toNat.toUInt64
  let n64 := f n64 shift64
  return n64.toLEByteArray.take k

def leftShift : Dtype -> ByteArray -> ByteArray -> Err ByteArray :=
  shift UInt64.shiftLeft

def leftShift! (dtype : Dtype) (bits : ByteArray) (shiftAmount : ByteArray) : ByteArray :=
  get! $ leftShift dtype bits shiftAmount

-- logical shift, no 1-extension
def rightShift : Dtype -> ByteArray -> ByteArray -> Err ByteArray :=
  shift UInt64.shiftRight

def rightShift! (dtype : Dtype) (bits : ByteArray) (shiftAmount : ByteArray) : ByteArray :=
  get! $ rightShift dtype bits shiftAmount

section Bitwise

open scoped Iterator.PairLockStep

private def bitwiseBinop (f : UInt8 -> UInt8 -> UInt8) (xtyp : Dtype) (x : ByteArray) (ytyp : Dtype) (y : ByteArray) : Err ByteArray := do
  if xtyp.isFloat || ytyp.isFloat then throw "float types do not support bitwise ops" else
  if xtyp.itemsize != x.size || ytyp.itemsize != y.size then throw "byte size mismatch" else
  let n := x.size
  let m := y.size
  let xList := x.toList
  let yList := y.toList
  let (xList, yList) :=
    if n == m then (xList, yList)
    else if n < m then (xList ++ List.replicate (n - m) 0, yList)
    else (xList, yList ++ List.replicate (m - n) 0)
  let mut res : List UInt8 := []
  for (a, b) in (xList, yList) do
    res := f a b :: res
  return ByteArray.mk res.reverse.toArray

end Bitwise

def bitwiseAnd : Dtype -> ByteArray -> Dtype -> ByteArray -> Err ByteArray :=
  bitwiseBinop UInt8.land

def bitwiseAnd! (xtyp : Dtype) (x : ByteArray) (ytyp : Dtype) (y : ByteArray) : ByteArray :=
  get! $ bitwiseAnd xtyp x ytyp y

def bitwiseOr : Dtype -> ByteArray -> Dtype -> ByteArray -> Err ByteArray :=
  bitwiseBinop UInt8.lor

def bitwiseOr! (xtyp : Dtype) (x : ByteArray) (ytyp : Dtype) (y : ByteArray) : ByteArray :=
  get! $ bitwiseOr xtyp x ytyp y

def bitwiseXor : Dtype -> ByteArray -> Dtype -> ByteArray -> Err ByteArray :=
  bitwiseBinop UInt8.xor

def bitwiseXor! (xtyp : Dtype) (x : ByteArray) (ytyp : Dtype) (y : ByteArray) : ByteArray :=
  get! $ bitwiseXor xtyp x ytyp y

private def bitwiseUnop (f : UInt8 -> UInt8) (xtyp : Dtype) (x : ByteArray) : Err ByteArray := do
  if xtyp.isFloat then throw "float types do not support bitwise ops" else
  if xtyp.itemsize != x.size then throw "byte size mismatch" else
  return ByteArray.mk (x.data.map f)

def bitwiseNot : Dtype -> ByteArray -> Err ByteArray :=
  bitwiseUnop UInt8.complement

def bitwiseNot! (xtyp : Dtype) (x : ByteArray) : ByteArray :=
  get! $ bitwiseNot xtyp x

private def closeEnough64 (x y : ByteArray) (err : Float := 0.000001) : Err Bool := do
  let x <- Dtype.float64.byteArrayToFloat64 x
  let y <- Dtype.float64.byteArrayToFloat64 y
  return (x - y).abs < err

private def closeEnough64! (x y : ByteArray) (err : Float := 0.000001) : Bool :=
  get! $ closeEnough64 x y err

private def closeEnough32 (x y : ByteArray) (err : Float32 := 0.000001) : Err Bool := do
  let x <- Dtype.float32.byteArrayToFloat32 x
  let y <- Dtype.float32.byteArrayToFloat32 y
  return (x - y).abs < err

private def closeEnough32! (x y : ByteArray) (err : Float32 := 0.000001) : Bool :=
  get! $ closeEnough32 x y err

-- tan = sin/cos is bitwise accurate at for float64 on my mac, but it failed on GH Actions.
#guard
  let typ := Dtype.float64
  let n : Float := 1.1
  let arr := n.toLEByteArray
  let sin := typ.sin! arr
  let cos := typ.cos! arr
  let div := typ.div! sin cos
  let tan := typ.tan! arr
  closeEnough64! div tan

-- The equality is not true for fp32, which loses more bits to rounding error
-- TODO: Improve this by finding the index of the first different bit.
#guard
  let typ := Dtype.float32
  let n : Float32 := 1.1
  let arr := n.toLEByteArray
  let sin := typ.sin! arr
  let cos := typ.cos! arr
  let div := typ.div! sin cos
  let tan := typ.tan! arr
  closeEnough32! div tan

-- log (e ^ x) = x is bitwise accurate at float64, at least for the numbers we try
#guard
  let typ := Dtype.float64
  let n : Float := 1.1
  let arr := n.toLEByteArray
  let exp := typ.exp! arr
  let log := typ.log! exp
  closeEnough64! log arr

-- log (e ^ x) = x is also bitwise accurate at float32, at least for the numbers we try
#guard
  let typ := Dtype.float32
  let n : Float32 := 1.1
  let arr := n.toLEByteArray
  let exp := typ.exp! arr
  let log := typ.log! exp
  closeEnough32! log arr

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

#guard Dtype.uint8.leftShift! (ByteArray.mk #[10]) (ByteArray.mk #[1]) == ByteArray.mk #[20]
#guard Dtype.int8.leftShift! (ByteArray.mk #[10]) (ByteArray.mk #[1]) == ByteArray.mk #[20]
#guard Dtype.uint16.leftShift! (ByteArray.mk #[10, 0]) (ByteArray.mk #[1]) == ByteArray.mk #[20, 0]
#guard Dtype.int16.leftShift! (ByteArray.mk #[10, 0]) (ByteArray.mk #[1]) == ByteArray.mk #[20, 0]
#guard Dtype.uint32.leftShift! (ByteArray.mk #[10, 0, 0, 0]) (ByteArray.mk #[1]) == ByteArray.mk #[20, 0, 0, 0]
#guard Dtype.int32.leftShift! (ByteArray.mk #[10, 0, 0, 0]) (ByteArray.mk #[1]) == ByteArray.mk #[20, 0, 0, 0]
#guard Dtype.uint64.leftShift! (ByteArray.mk #[10, 0, 0, 0, 0, 0, 0, 0]) (ByteArray.mk #[1]) == ByteArray.mk #[20, 0, 0, 0, 0, 0, 0, 0]
#guard Dtype.int64.leftShift! (ByteArray.mk #[10, 0, 0, 0, 0, 0, 0, 0]) (ByteArray.mk #[1]) == ByteArray.mk #[20, 0, 0, 0, 0, 0, 0, 0]
#guard Dtype.int8.leftShift! (ByteArray.mk #[0xFF]) (ByteArray.mk #[1]) == ByteArray.mk #[0xFE]
#guard !(Dtype.bool.leftShift (ByteArray.mk #[0x1]) (ByteArray.mk #[1])).isOk
#guard !(Dtype.uint16.leftShift (ByteArray.mk #[0x1]) (ByteArray.mk #[1])).isOk
#guard !(Dtype.float32.leftShift (ByteArray.mk #[0x1, 0, 0, 0]) (ByteArray.mk #[1])).isOk
#guard !(Dtype.float64.leftShift (ByteArray.mk #[0x1, 0, 0, 0, 0, 0, 0, 0]) (ByteArray.mk #[1])).isOk

end Test

end Dtype
end TensorLib
