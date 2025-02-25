/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/

import TensorLib.Common

namespace TensorLib
namespace Dtype

/-! The subset of types NumPy supports that we care about -/
inductive Name where
| bool
| int8
| int16
| int32
| int64
| uint8
| uint16
| uint32
| uint64
| float16
| float32
| float64
deriving BEq, Repr, Inhabited

namespace Name

instance : ToString Name where
  toString x := ((repr x).pretty.splitOn ".").getLast!

def isOneByte (x : Name) : Bool := match x with
| bool | int8 | uint8 => true
| _ => false

def isMultiByte (x : Name) : Bool := ! x.isOneByte

def isInt (x : Name) : Bool := match x with
| int8 | int16 | int32 | int64 => true
| _ => false

def isUint (x : Name) : Bool := match x with
| uint8 | uint16 | uint32 | uint64 => true
| _ => false

def isIntLike (x : Name) : Bool := x.isInt || x.isUint

--! Number of bytes used by each element of the given dtype
def itemsize (x : Name) : Nat := match x with
| float64 | int64 | uint64 => 8
| float32 | int32 | uint32 => 4
| float16 | int16 | uint16 => 2
| bool | int8 | uint8 => 1

end Name
end Dtype

structure Dtype where
  name : Dtype.Name
  order : ByteOrder
deriving BEq, Repr, Inhabited

namespace Dtype

def bool : Dtype := Dtype.mk .bool .oneByte
def int8 : Dtype := Dtype.mk .int8 .oneByte
def int16 : Dtype := Dtype.mk .int16 .littleEndian
def int32 : Dtype := Dtype.mk .int32 .littleEndian
def int64 : Dtype := Dtype.mk .int64 .littleEndian
def uint8 : Dtype := Dtype.mk .uint8 .oneByte
def uint16 : Dtype := Dtype.mk .uint16 .littleEndian
def uint32 : Dtype := Dtype.mk .uint32 .littleEndian
def uint64 : Dtype := Dtype.mk .uint64 .littleEndian
def float16 : Dtype := Dtype.mk .float16 .littleEndian
def float32 : Dtype := Dtype.mk .float32 .littleEndian
def float64 : Dtype := Dtype.mk .float64 .littleEndian

def make (name : Name) (order : ByteOrder) : Err Dtype := match order with
| .oneByte => if name.isOneByte then .ok $ mk name order else .error "illegal dtype"
| .littleEndian | .bigEndian => if name.isMultiByte then .ok $ mk name order else .error "illegal dtype"

def byteOrderOk (dtype : Dtype) : Prop := !dtype.name.isMultiByte || (dtype.name.isMultiByte && dtype.order.isMultiByte)

theorem makeOk (name : Name) (order : ByteOrder) : match make name order with
| .ok dtype => dtype.byteOrderOk
| .error _ => true := by
  unfold make byteOrderOk Name.isMultiByte Name.isOneByte
  cases name <;> cases order <;> simp

def itemsize (dtype : Dtype) := dtype.name.itemsize

def sizedStrides (dtype : Dtype) (s : Shape) : Strides := List.map (fun x => x * dtype.itemsize) s.unitStrides

/-
For the signed types below like int8, we could allow up to 0xFF..., but then values above 0x7F...
give negative numbers, which is confusing.
-/

def byteArrayOfNat (dtype : Dtype) (n : Nat) : Err ByteArray := match dtype.name with
| .bool => if n <= 1 then .ok (BV8.ofNat n).toByteArray else .error "Nat out of bounds for bool"
| .uint8 => if n <= 0xFF then .ok (BV8.ofNat n).toByteArray else .error "Nat out of bounds for uint8"
| .int8 => if n <= 0x7F then .ok [(Int8.ofNat n).toUInt8].toByteArray else .error "Nat out of bounds for int8"
| .uint16 => if n <= 0xFFFF then .ok $ BV16.toByteArray n.toUInt16.toBitVec dtype.order else .error "Nat out of bounds for uint16"
| .int16 => if n <= 0x7FFF then .ok $ BV16.toByteArray n.toInt16.toBitVec dtype.order else .error "Nat out of bounds for int16"
| .uint32 => if n <= 0xFFFFFFFF then .ok $ BV32.toByteArray n.toUInt32.toBitVec dtype.order else .error "Nat out of bounds for uint32"
| .int32 => if n <= 0x7FFFFFFF then .ok $ BV32.toByteArray n.toInt32.toBitVec dtype.order else .error "Nat out of bounds for int32"
| .uint64 => if n <= 0xFFFFFFFFFFFFFFFF then .ok $ BV64.toByteArray n.toUInt64.toBitVec dtype.order else .error "Nat out of bounds for uint64"
| .int64 => if n <= 0x7FFFFFFFFFFFFFFF then .ok $ BV64.toByteArray n.toInt64.toBitVec dtype.order else .error "Nat out of bounds for int64"
| .float16
| .float32 => .error "Sub-word floats are not yet supported by lean"
| .float64 => .error "Float not yet supported"

def byteArrayOfNat! (dtype : Dtype) (n : Nat) : ByteArray := get! $ byteArrayOfNat dtype n

def byteArrayToNat (dtype : Dtype) (arr : ByteArray) : Err Nat :=
  if dtype.itemsize != arr.size then .error "byte size mismatch"
  else .ok $ dtype.order.bytesToNat arr

def byteArrayToNat! (dtype : Dtype) (arr : ByteArray) : Nat := get! $ byteArrayToNat dtype arr

private def byteArrayToNatRoundTrip (dtype : Dtype) (n : Nat) : Bool :=
  let res := do
    let arr <- dtype.byteArrayOfNat n
    let n' <- dtype.byteArrayToNat arr
    return n == n'
  res.toOption.getD false

#guard uint8.byteArrayToNatRoundTrip 0
#guard uint8.byteArrayToNatRoundTrip 5
#guard uint8.byteArrayToNatRoundTrip 255
#guard !uint8.byteArrayToNatRoundTrip 256

def byteArrayOfInt (dtype : Dtype) (n : Int) : Err ByteArray := match dtype.name with
| .bool => if 0 <= n && n <= 1 then .ok (BV8.ofNat n.toNat).toByteArray else .error "out of bounds"
| .uint8
| .int8 => if -0x80 <= n && n <= 0x7F then .ok [n.toInt8.toUInt8].toByteArray else .error "out of bounds"
| .uint16
| .int16 => if -0x8000 <= n && n <= 0x7FFF then .ok $ BV16.toByteArray n.toInt16.toBitVec dtype.order else .error "out of bounds"
| .uint32
| .int32 => if -0x80000000 <= n && n <= 0x7FFFFFFF then .ok $ BV32.toByteArray n.toInt32.toBitVec dtype.order else .error "out of bounds"
| .uint64
| .int64 => if -0x8000000000000000 <= n && n <= 0x7FFFFFFFFFFFFFFF then .ok $ BV64.toByteArray n.toInt64.toBitVec dtype.order else .error "out of bounds"
| .float16
| .float32 => .error "Sub-word floats are not yet supported by lean"
| .float64 => .error "Float not yet supported"

def byteArrayOfInt! (dtype : Dtype) (n : Int) : ByteArray := get! $ byteArrayOfInt dtype n

def byteArrayToInt (dtype : Dtype) (arr : ByteArray) : Err Int :=
  if dtype.itemsize != arr.size then .error "byte size mismatch"
  else .ok $ dtype.order.bytesToInt arr

def byteArrayToInt! (dtype : Dtype) (arr : ByteArray) : Int := get! $ byteArrayToInt dtype arr

private def byteArrayToIntRoundTrip (dtype : Dtype) (n : Int) : Bool :=
  let res := do
    let arr <- dtype.byteArrayOfInt n
    let n' <- dtype.byteArrayToInt arr
    return n == n'
  res.toOption.getD false

#guard int8.byteArrayToIntRoundTrip 0
#guard int8.byteArrayToIntRoundTrip 5
#guard int8.byteArrayToIntRoundTrip 127
#guard !int8.byteArrayToIntRoundTrip 255

def byteArrayToFloat (dtype : Dtype) (arr : ByteArray) : Err Float := match dtype.name with
| .float64 =>
  if arr.size != 8 then .error "byte size mismatch" else
  match dtype.order with
  | .littleEndian => .ok $ Float.ofBits arr.toUInt64LE! -- toUInt64LE! is ok here because we already checked the size
  | .bigEndian => .ok $ Float.ofBits arr.toUInt64BE! -- toUInt64BE! is ok here because we already checked the size
  | .oneByte => impossible "Illegal dtype. Creation shouldn't have been possible"
| .float16 | .float32 => .error "Unsupported float type. Requires Lean support."
| _ => .error "Illegal type conversion"

def byteArrayToFloat! (dtype : Dtype) (arr : ByteArray) : Float := get! $ byteArrayToFloat dtype arr

def byteArrayOfFloat (dtype : Dtype) (f : Float) : Err ByteArray := match dtype.name with
| .float64 => .ok $ BV64.toByteArray f.toBits.toBitVec dtype.order
| .float16 | .float32 => .error "Unsupported float type. Requires Lean support."
| _ => .error "Illegal type conversion"

def byteArrayOfFloat! (dtype : Dtype) (f : Float) : ByteArray := get! $ byteArrayOfFloat dtype f

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
#guard !float16.byteArrayToFloatRoundTrip 0
#guard !float32.byteArrayToFloatRoundTrip 0

private def floatToByteArray (f : Float) : Array UInt8 :=
  (BV64.toByteArray f.toUInt64.toBitVec ByteOrder.littleEndian).data

private def byteToFloatArray (b : UInt8) : Array UInt8 := floatToByteArray (Float.ofNat b.toNat)

/-
64-bit IEEE-754 floats have 52-bit significand with an implicit leading 1. This means we can
represent every int in [-2^^53:2^^53]  (0 obviously as well, with a different interpretation of the bits).
For example, if we had a 3-bit significand and 3-bit exponent we could represent the following contiguous table.

    N : sign exp sig
    0 : 0    000 0000 -- 0 is denormalized; an all-0 exponent means there's no bias
    1 : 0    011 0000 -- bias is 0b11 = 3, assumes a leading 1, so (1 + 0) * 2^(3 - 3) = 1
    2 : 0    100 0000 -- (1 + 0) * 2^(4 - 3) = 2
    3 : 0    100 1000 -- (1 + .1) * 2^(4 - 3) = 0b11 = 3
    ...
             101 1111 -- (1 + .1111) * 2^(5 - 3) = 0b11111 = 31
            Note: can't use 111 for exponent, which is used for ±infty and nan.
            We also can't use 110 because we'd skip some not exactly representable ints.

    Negative values of N just switch the sign bit.

    In 64-bit floats, it is 2^53 - 1. This is even named in JavaScript as Number.MAX_SAFE_INTEGER
-/
private def natToFloatByteArray (n : Nat) : Err (Array UInt8) :=
  if Nat.pow 2 53 <= n then .error "overflow" else .ok $ floatToByteArray n.toFloat

private def unsignedLEByteArrayToNat (arr : Array UInt8) : Nat := Id.run do
  let mut res : Nat := 0
  let mut pow : Nat := 0
  for byte in arr do
    res := res + byte.toNat * Nat.pow 2 pow
    pow := pow + 1
  return res

private def unsignedBEByteArrayToNat (arr : Array UInt8) : Nat := unsignedLEByteArrayToNat arr.reverse

#guard unsignedLEByteArrayToNat #[1, 0, 1, 1] == 13
#guard unsignedBEByteArrayToNat #[1, 0, 1, 1] == 11

def isInt (dtype : Dtype) : Bool := dtype.name.isInt
def isUint (dtype : Dtype) : Bool := dtype.name.isUint
def isIntLike (dtype : Dtype) : Bool := dtype.isInt || dtype.isUint

def add (dtype : Dtype) (x y : ByteArray) : Err ByteArray :=
  if dtype.itemsize != x.size || dtype.itemsize != y.size then .error "add: byte size mismatch" else
  match dtype.name with
  | .uint8 | .uint16 | .uint32 | .uint64 => do
    let x <- byteArrayToNat dtype x
    let y <- byteArrayToNat dtype y
    byteArrayOfNat dtype (x + y)
  | .int8 | .int16| .int32 | .int64 => do
    let x <- byteArrayToInt dtype x
    let y <- byteArrayToInt dtype y
    byteArrayOfInt dtype (x + y)
  | .float64 => do
    let x <- byteArrayToFloat dtype x
    let y <- byteArrayToFloat dtype y
    byteArrayOfFloat dtype (x + y)
  | .bool | .float16 | .float32 => .error s!"`add` not supported at type ${dtype.name}"

def sub (dtype : Dtype) (x y : ByteArray) : Err ByteArray :=
  if dtype.itemsize != x.size || dtype.itemsize != y.size then .error "sub: byte size mismatch" else
  match dtype.name with
  | .uint8 | .uint16 | .uint32 | .uint64 => do
    let x <- byteArrayToNat dtype x
    let y <- byteArrayToNat dtype y
    byteArrayOfNat dtype (x - y)
  | .int8 | .int16| .int32 | .int64 => do
    let x <- byteArrayToInt dtype x
    let y <- byteArrayToInt dtype y
    byteArrayOfInt dtype (x - y)
  | .float64 => do
    let x <- byteArrayToFloat dtype x
    let y <- byteArrayToFloat dtype y
    byteArrayOfFloat dtype (x - y)
  | .bool | .float16 | .float32 => .error s!"`sub` not supported at type ${dtype.name}"

def mul (dtype : Dtype) (x y : ByteArray) : Err ByteArray :=
  if dtype.itemsize != x.size || dtype.itemsize != y.size then .error "mul: byte size mismatch" else
  match dtype.name with
  | .uint8 | .uint16 | .uint32 | .uint64 => do
    let x <- byteArrayToNat dtype x
    let y <- byteArrayToNat dtype y
    byteArrayOfNat dtype (x * y)
  | .int8 | .int16| .int32 | .int64 => do
    let x <- byteArrayToInt dtype x
    let y <- byteArrayToInt dtype y
    byteArrayOfInt dtype (x * y)
  | .float64 => do
    let x <- byteArrayToFloat dtype x
    let y <- byteArrayToFloat dtype y
    byteArrayOfFloat dtype (x * y)
  | .bool | .float16 | .float32 => .error s!"`mul` not supported at type ${dtype.name}"

def div (dtype : Dtype) (x y : ByteArray) : Err ByteArray :=
  if dtype.itemsize != x.size || dtype.itemsize != y.size then .error "div: byte size mismatch" else
  match dtype.name with
  | .uint8 | .uint16 | .uint32 | .uint64 => do
    let x <- byteArrayToNat dtype x
    let y <- byteArrayToNat dtype y
    byteArrayOfNat dtype (x / y)
  | .int8 | .int16| .int32 | .int64 => do
    let x <- byteArrayToInt dtype x
    let y <- byteArrayToInt dtype y
    byteArrayOfInt dtype (x / y)
  | .float64 => do
    let x <- byteArrayToFloat dtype x
    let y <- byteArrayToFloat dtype y
    byteArrayOfFloat dtype (x / y)
  | .bool | .float16 | .float32 => .error s!"`div` not supported at type ${dtype.name}"

/-
This works for int/uint/bool/float. Keep an eye out when we start implementing unusual floating point types.
-/
def zero (dtype : Dtype) : ByteArray := ByteArray.mk $ (List.replicate dtype.itemsize (0 : UInt8)).toArray

end Dtype
end TensorLib
