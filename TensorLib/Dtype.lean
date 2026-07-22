/-
Copyright TensorLib Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
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
| float8_e4m3
| float8_e5m2
| float16
| bfloat16
| float32
| float64
deriving BEq, Repr, Inhabited, DecidableEq


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
  float8_e4m3,
  float8_e5m2,
  float16,
  bfloat16,
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
  | float8_e4m3 => "float8_e4m3fn"
  | float8_e5m2 => "float8_e5m2" -- no fn since e5m2 has infinity
  | float16 => "float16"
  | bfloat16 => "bfloat16"
  | float32 => "float32"
  | float64 => "float64"


def isOneByte (x : Dtype) : Bool := match x with
| bool | int8 | uint8 | float8_e4m3 | float8_e5m2 => true
| _ => false

def isMultiByte (x : Dtype) : Bool := ! x.isOneByte

def isInt (x : Dtype) : Bool := match x with
| int8 | int16 | int32 | int64 => true
| _ => false

def isUint (x : Dtype) : Bool := match x with
| uint8 | uint16 | uint32 | uint64 => true
| _ => false

def isIntLike (x : Dtype) : Bool := x.isInt || x.isUint

-- when we cast +inf -> int32, we can saturate to INT32_MAX (2147483647)
-- the max value that int32 can hold.
def intMin (x : Dtype) : Int := match x with
| .int8   => -128
| .int16  => -32768
| .int32  => -2147483648
| .int64  => -9223372036854775808
| _  => 0

def intMax (x : Dtype) : Int := match x with
| .int8   => 127
| .int16  => 32767
| .int32  => 2147483647
| .int64  => 9223372036854775807
| .uint8  => 255
| .uint16 => 65535
| .uint32 => 4294967295
| .uint64 => 18446744073709551615
| _ => 0

-- Added float16 and bfloat16 so bitwise op know to reject it
def isFloat (x : Dtype) : Bool := match x with
| .float16 | .bfloat16 | .float32 | .float64 | .float8_e4m3 | float8_e5m2 => true
| _ => false

--! Number of bytes used by each element of the given dtype
def itemsize (x : Dtype) : Nat := match x with
| float64 | int64 | uint64 => 8
| float32 | int32 | uint32 => 4
| bfloat16 | float16 | int16 | uint16 => 2
| bool | int8 | uint8 | float8_e4m3 | float8_e5m2 => 1

-- Previously this was inline in join with a recursive swap,
-- but adding more fp8 types made the match too large. Lean needs to prove
-- termination and derive unfold equations for recursive functions, and
-- both timed out on the large match (which is monotonically growing w the dtypes).
-- Splitting into a non-recursive helper eliminates the recursion so Lean skips those expensive steps.
private def joinOrdered (x y : Dtype) : Option Dtype :=
  match x, y with
  | .float16, .float32 => float32
  | .float16, .float64 => float64
  | .float16, .int16 => float32
  | .float16, .uint16 => float32
  | .float16, .int32 => float64
  | .float16, .uint32 => float64
  | .float16, .int64 => float64
  | .float16, .uint64 => float64
  | .float16, .bfloat16 => none
  | .bfloat16, .float16 => none
  | .bfloat16, .float32 => float32
  | .bfloat16, .float64 => float64
  | .bfloat16, _ => none
  | .float8_e4m3, .float32 => float32
  | .float8_e4m3, .float64 => float64
  | .float8_e4m3, .bool
  | .float8_e4m3, .int8
  | .float8_e4m3, .uint8 => float8_e4m3
  | .float8_e4m3, _ => none
  | .float8_e5m2, .float32 => float32
  | .float8_e5m2, .float64 => float64
  | .float8_e5m2, .float16 => float32
  | .float8_e5m2, .int16 => float32
  | .float8_e5m2, .uint16 => float32
  | .float8_e5m2, .int32 => float64
  | .float8_e5m2, .uint32 => float64
  | .float8_e5m2, .int64 => float64
  | .float8_e5m2, .uint64 => float64
  | .float8_e5m2, .bool => float8_e5m2
  | .float8_e5m2, .int8 => float8_e5m2
  | .float8_e5m2, .uint8 => float8_e5m2
  | .float8_e5m2, _ => none
  | .float32, .float64 => float64
  | .float32, _
  | _, .float32 => none
  | .float64, _
  | _, .float64 => none
  | .bool, _ => y
  | .int8, .uint8
  | .uint8, .int8 => int16
  | .int8, .uint16 => int32
  | .int8, .uint32 => int64
  | .int8, .uint64 => float64
  | .int8, .bool => int8
  | .uint8, .bool => uint8
  | .int8, _
  | .uint8, _ => y
  | .int16, .uint16
  | .uint16, .int16 => int32
  | .int16, .bfloat16 => none
  | .int16, .float16 => float32
  | .int16, .int32 => int32
  | .int16, .int64 => int64
  | .int16, .uint32 => int64
  | .int16, .uint64 => float64
  | .int16, _ => none
  | .uint16, .bfloat16 => none
  | .uint16, .float16 => float32
  | .uint16, _ => y
  | .int32, .uint32
  | .uint32, .int32 => int64
  | .int32, .uint64 => float64
  | .int32, _
  | .uint32, _ => y
  | .uint64, .int64
  | .int64, .uint64 => float64
  | .int64, _
  | .uint64, _ => none
  -- catch all for any remaining pairs not listed since they are unreachable (join handles eq and swaps) or ones that cannot be promoted
  | _, _ => none

def join (x y : Dtype) : Option Dtype :=
  if x = y then x else if x.itemsize > y.itemsize then joinOrdered y x else joinOrdered x y


-- Can we cast from one dtype to another without losing information
def lossless (fromDtype toDtype : Dtype) : Bool := match fromDtype, toDtype with
| .bool, _ => true
| _, .bool => false
| .int8, .int8
| .int8, .int16
| .int8, .int32
| .int8, .int64
| .int8, .float16
| .int8, .bfloat16
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
| .uint8, .float16
| .uint8, .bfloat16
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
| .float16, .float16
| .float16, .float32
| .float16, .float64 => true
| .float16, _ => false
| .bfloat16, .bfloat16
| .bfloat16, .float32
| .bfloat16, .float64 => true
| .bfloat16, _ => false
| .float8_e4m3, .float8_e4m3
| .float8_e4m3, .float16
| .float8_e4m3, .bfloat16
| .float8_e4m3, .float32
| .float8_e4m3, .float64 => true
| .float8_e4m3, _ => false
| .float8_e5m2, .float8_e5m2
| .float8_e5m2, .float16
| .float8_e5m2, .bfloat16
| .float8_e5m2, .float32
| .float8_e5m2, .float64 => true
| .float8_e5m2, _ => false
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
| .float8_e4m3 => maxSafeNatForFloat8e4m3
| .float8_e5m2 => maxSafeNatForFloat8e5m2
| .float16 => maxSafeNatForFloat16
| .bfloat16 => maxSafeNatForBFloat16
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
| .float8_e4m3 => some (-maxSafeNatForFloat8e4m3)
| .float8_e5m2 => some (-maxSafeNatForFloat8e5m2)
| .float16 => some (-maxSafeNatForFloat16)
| .bfloat16 => some (-maxSafeNatForBFloat16)
| .float32 => some (-maxSafeNatForFloat32)
| .float64 => some (-maxSafeNatForFloat64)

private def canCastFromInt (dtype : Dtype) (n : Int) : Bool :=
  if n < 0 then dtype.minSafeInt.getD n <= n
  else n <= dtype.maxSafeNat.getD n.toNat

def sizedStrides (dtype : Dtype) (s : Shape) : Strides := List.map (fun x => x * dtype.itemsize) s.unitStrides

-- Decode 1-byte fp8_e4m3 to fp32
-- Prevent having to repeat size check at call sites
def decodeFloat8E4M3 (arr : ByteArray) : Err Float32 :=
  if arr.size != 1 then .error "decoder: expected 1 byte for float8_e4m3"
  else .ok (arr.data[0]!.toFloat32FromFloat8E4M3)

-- Encode Float32 to 1-byte fp8_e4m3
private def encodeFloat8E4M3 (f : Float32) : ByteArray :=
  ByteArray.mk #[f.toFloat8E4M3Bits]

-- Decode 1-byte float8_e5m2 to Float32.
-- Centralizes the size check so callers don't need inline guards.
def decodeFloat8E5M2 (arr : ByteArray) : Err Float32 :=
  if arr.size != 1 then .error "decoder: expected 1 byte for float8_e5m2" else .ok (arr.data[0]!.toFloat32FromFloat8E5M2)

-- Encode Float32 to 1-byte float8_e5m2.
private def encodeFloat8E5M2 (f : Float32) : ByteArray :=
  ByteArray.mk #[f.toFloat8E5M2Bits]

def byteArrayOfNatOverflow (dtype : Dtype) (n : Nat) : ByteArray := match dtype with
| .bool => toLEByteArray (if n == 0 then 0 else 1).toUInt8
| .uint8 => toLEByteArray n.toUInt8
| .int8 => toLEByteArray n.toInt8
| .uint16 => toLEByteArray n.toUInt16
| .int16 => toLEByteArray n.toInt16
| .uint32 => toLEByteArray n.toUInt32
| .int32 => toLEByteArray n.toInt32
| .uint64 => toLEByteArray n.toUInt64
| .int64 => toLEByteArray n.toInt64
| .float8_e4m3 => encodeFloat8E4M3 n.toFloat32
| .float8_e5m2 => encodeFloat8E5M2 n.toFloat32
| .float16 => toLEByteArray n.toFloat32.toFloat16Bits
| .bfloat16 => toLEByteArray n.toFloat32.toBFloat16Bits
| .float32 => toLEByteArray n.toFloat32
| .float64 => toLEByteArray n.toFloat

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
-- Saturating fp32 → Nat, keyed on target unsigned dtype.
-- +inf -> uintMax, -inf/NaN -> 0 for all sizes.
-- Finite overflow: saturate for uint32/uint64 (matches numpy),
-- wrap for uint8/uint16 (also matches numpy's C cast behavior).
-- Negative finite: wrap for uint8/uint16, clamp to 0 for uint32/uint64.
private def saturatingNatOfFloat32 (dtype : Dtype) (f : Float32) : Nat :=
  if f.isNaN then 0
  else if f.isPosInf then dtype.intMax.toNat
  else if f.isNegInf then 0
  else
    match dtype with
    | .uint32 | .uint64 =>
      if f <= 0 then 0
      else min f.toNat dtype.intMax.toNat
    | _ => f.toNat

-- Same for fp64
private def saturatingNatOfFloat64 (dtype : Dtype) (f : Float) : Nat :=
  if f.isNaN then 0
  else if f.isPosInf then dtype.intMax.toNat
  else if f.isNegInf then 0
  else
    match dtype with
    | .uint32 | .uint64 =>
      if f <= 0 then 0
      else min f.toNat dtype.intMax.toNat
    | _ => f.toNat
-- Saturating fp32 to Int, depends on target integer dtype.
-- +inf -> intMax, -inf -> intMin, NaN -> 0
-- finite overflow: satuate for int32/64 as per numpy
-- wrap for int8/16 (numpy)
private def saturatingIntOfFloat32 (dtype : Dtype) (f : Float32) : Int :=
  if f.isNaN then 0
  else if f.isPosInf then dtype.intMax
  else if f.isNegInf then dtype.intMin
  else
    match dtype with
    | .int32 | .int64 => min (max f.toInt dtype.intMin) dtype.intMax
    | _ => f.toInt

-- Saturating fp64 to Int
private def saturatingIntOfFloat64 (dtype : Dtype) (f : Float) : Int :=
  if f.isNaN then 0
  else if f.isPosInf then dtype.intMax
  else if f.isNegInf then dtype.intMin
  else
    match dtype with
    | .int64 | .uint64 => min (max f.toInt dtype.intMin) dtype.intMax
    | _ => f.toInt

private def byteArrayOfIntOverflow (dtype : Dtype) (n : Int) : ByteArray := match dtype with
| .bool => toLEByteArray (UInt8.ofNat (if n == 0 then 0 else 1))
| .uint8 | .int8 => toLEByteArray n.toInt8
| .uint16 | .int16 => toLEByteArray n.toInt16
| .uint32 | .int32 => toLEByteArray n.toInt32
| .uint64 | .int64 => toLEByteArray n.toInt64
| .float8_e4m3 => encodeFloat8E4M3 n.toFloat32
| .float8_e5m2 => encodeFloat8E5M2 n.toFloat32
| .float16 => toLEByteArray n.toFloat32.toFloat16Bits
| .bfloat16 => toLEByteArray n.toFloat32.toBFloat16Bits
| .float32 => toLEByteArray n.toFloat32
| .float64 => toLEByteArray n.toFloat64


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
| .float64 => .ok $ toLEByteArray f
| _ => .error "Illegal type conversion"

private def byteArrayOfFloat64! (dtype : Dtype) (f : Float) : ByteArray := get! $ byteArrayOfFloat64 dtype f

def byteArrayOfFloat32 (dtype : Dtype) (f : Float32) : Err ByteArray := match dtype with
| .float32 => .ok $ toLEByteArray f
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

-- Read 2 bytes as fp 16, upcast to fp32 for computations
def byteArrayToFloat16 (dtype : Dtype)(arr : ByteArray) : Err Float32 := match dtype with
  | .float16 => arr.toUInt16LE.map UInt16.toFloat32FromFloat16
  | _ => .error "Illegal type conversion"

-- helper functions for fp16 and bf16 arithmetic to reduce duplicate code
-- Decode/ encode 2 byte float to float32 by calling the conversion kernel directly
private def decodeFloat16OrBFloat16 (dtype : Dtype) (arr : ByteArray) : Err Float32 := match dtype with
  | .float16 => arr.toUInt16LE.map UInt16.toFloat32FromFloat16
  | .bfloat16 => arr.toUInt16LE.map UInt16.toFloat32FromBFloat16
  | _ => .error "decoder: expected float16 or bfloat16"

private def encodeFloat16OrBFloat16 (dtype : Dtype) (f : Float32) : Err ByteArray := match dtype with
  | .float16 => .ok (toLEByteArray f.toFloat16Bits)
  | .bfloat16 => .ok (toLEByteArray f.toBFloat16Bits)
  | _ => .error "encoder: expected float16 or bfloat16"

private def byteArrayToFloat16RoundTrip (dtype : Dtype) (f : Float32) : Bool :=
  let res := do
    let arr <- encodeFloat16OrBFloat16 dtype f
    let f' <- decodeFloat16OrBFloat16 dtype arr
    return f == f'
  res.toOption.getD false

#guard float16.byteArrayToFloat16RoundTrip 0
#guard float16.byteArrayToFloat16RoundTrip 1.5
#guard float16.byteArrayToFloat16RoundTrip 42
#guard float16.byteArrayToFloat16RoundTrip (-0)

-- Decode bf16 bytes to Float32. bf16 shares fp32's exponent, so just pad mantissa with 0's.
def byteArrayToBFloat16 (dtype : Dtype) (arr : ByteArray) : Err Float32 := match dtype with
  | .bfloat16 => arr.toUInt16LE.map UInt16.toFloat32FromBFloat16
  | _ => .error "Illegal type conversion"

-- roundtrip helper: encode fp32 -> bf16 -> fp32
-- if output is same as input we know the encoder and decoder are consistent
private def byteArrayToBFloat16RoundTrip (dtype : Dtype) (f : Float32) : Bool :=
  let res := do
    let arr <- encodeFloat16OrBFloat16 dtype f -- fp32 to 2 bf16 bytes
    let f' <- decodeFloat16OrBFloat16 dtype arr-- back to fp32
    return f == f'
  res.toOption.getD false -- return false if this returned an error

#guard bfloat16.byteArrayToBFloat16RoundTrip 0
#guard bfloat16.byteArrayToBFloat16RoundTrip 1.5
#guard bfloat16.byteArrayToBFloat16RoundTrip 42
#guard bfloat16.byteArrayToBFloat16RoundTrip (-0)
#guard bfloat16.byteArrayToBFloat16RoundTrip 256



-- Add produces the IEEE754 result because fp32 (p=24) satisfies the innocuous double rounding condition (theoreom 20)
-- https://hal.science/hal-01091186v1/document
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
  | .float8_e4m3 => do
    let x <- decodeFloat8E4M3 x
    let y <- decodeFloat8E4M3 y
    return encodeFloat8E4M3 (x + y)
  | .float8_e5m2 => do
    let x <- decodeFloat8E5M2 x
    let y <- decodeFloat8E5M2 y
    return encodeFloat8E5M2 (x + y)
  | .float16
  | .bfloat16 => do
    let x <- dtype.decodeFloat16OrBFloat16 x
    let y <- dtype.decodeFloat16OrBFloat16 y
    dtype.encodeFloat16OrBFloat16 (x + y)
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
  | .float8_e4m3 => do
    let x <- decodeFloat8E4M3 x
    let y <- decodeFloat8E4M3 y
    return encodeFloat8E4M3 (x - y)
  | .float8_e5m2 => do
    let x <- decodeFloat8E5M2 x
    let y <- decodeFloat8E5M2 y
    return encodeFloat8E5M2 (x - y)
  | .float16
  | .bfloat16 => do
    let x <- dtype.decodeFloat16OrBFloat16 x
    let y <- dtype.decodeFloat16OrBFloat16 y
    dtype.encodeFloat16OrBFloat16 (x - y)
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
  | .float8_e4m3 => do
    let x <- decodeFloat8E4M3 x
    let y <- decodeFloat8E4M3 y
    return encodeFloat8E4M3 (x * y)
  | .float8_e5m2 => do
    let x <- decodeFloat8E5M2 x
    let y <- decodeFloat8E5M2 y
    return encodeFloat8E5M2 (x * y)
  | .float16
  | .bfloat16 => do
    let x <- dtype.decodeFloat16OrBFloat16 x
    let y <- dtype.decodeFloat16OrBFloat16 y
    dtype.encodeFloat16OrBFloat16 (x * y)
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
  | .float8_e4m3 => do
    let x <- decodeFloat8E4M3 x
    let y <- decodeFloat8E4M3 y
    return encodeFloat8E4M3 (x / y)
  | .float8_e5m2 => do
    let x <- decodeFloat8E5M2 x
    let y <- decodeFloat8E5M2 y
    return encodeFloat8E5M2 (x / y)
  | .float16
  | .bfloat16 => do
    let x <- dtype.decodeFloat16OrBFloat16 x
    let y <- dtype.decodeFloat16OrBFloat16 y
    dtype.encodeFloat16OrBFloat16 (x / y)
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
  | .float8_e4m3 => do
    let f <- decodeFloat8E4M3 x
    return encodeFloat8E4M3 f.abs
  | .float8_e5m2 => do
    let f <- decodeFloat8E5M2 x
    return encodeFloat8E5M2 f.abs
  | .float16
  | .bfloat16 => do
    let x <- dtype.decodeFloat16OrBFloat16 x
    dtype.encodeFloat16OrBFloat16 x.abs
  | .float32 =>
    let x <- dtype.byteArrayToFloat32 x
    dtype.byteArrayOfFloat32 x.abs
  | .float64 => do
    let x <- dtype.byteArrayToFloat64 x
    dtype.byteArrayOfFloat64 x.abs
  | .bool => return x

-- abs rejects wrong-sized input for fp8_e4m3
#guard (Dtype.abs .float8_e4m3 (ByteArray.mk #[])).isOk == false -- 0 bytes (should be at least 1 byte for fp8_e4m3)
#guard (Dtype.abs .float8_e4m3 (ByteArray.mk #[1, 2])).isOk == false -- 2 bytes (too many for a 1 byte type)

def abs! (dtype : Dtype) (x : ByteArray) : ByteArray := get! $ abs dtype x

-- copy pasted from below due to call in next function
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
| float8_e4m3 => do
  let f <- decodeFloat8E4M3 x
  return f == 0
| float8_e5m2 => do
  let f <- decodeFloat8E5M2 x
  return f == 0
| float16
| bfloat16 => do
  let f <- dtype.decodeFloat16OrBFloat16 x
  return f == 0
| float32 => do
  let f <- Float32.ofLEByteArray x
  return f == 0
| float64 => do
  let f <- Float.ofLEByteArray x
  return f == 0

-- isZero rejects wrong-sized input for float8_e4m3
#guard (Dtype.isZero .float8_e4m3 (ByteArray.mk #[])).isOk == false
#guard (Dtype.isZero .float8_e4m3 (ByteArray.mk #[1, 2])).isOk == false

def castOverflow (fromDtype : Dtype) (data : ByteArray) (toDtype : Dtype) : Err ByteArray :=
  if fromDtype == toDtype then return data else
    match fromDtype, toDtype with
    -- For floats use isZero so -0 is correctly handled.
    -- A raw byte check would treat -0.0 as !0 since sign bit is nonzero
    | _, bool =>
      if fromDtype.isFloat then do
        let isZ <- fromDtype.isZero data
        return ByteArray.mk #[if isZ then 0 else 1]
      else
        return ByteArray.mk #[if data.data.all fun x => x == 0 then 0 else 1]
    | .bool, _ | .uint8, _ | .uint16, _ | .uint32, _ | .uint64, _ =>
      return toDtype.byteArrayOfNatOverflow data.toNat
    | .int8, _ | .int16, _ | .int32, _ | .int64, _ =>
      return toDtype.byteArrayOfIntOverflow data.toInt
    | .float32, .uint8 | .float32, .uint16 | .float32, .uint32 | .float32, .uint64 => do
      let f <- Float32.ofLEByteArray data
      return toDtype.byteArrayOfNatOverflow (saturatingNatOfFloat32 toDtype f)
    | .float32, .int8 | .float32, .int16 | .float32, .int32 | .float32, .int64 => do
      let f <- Float32.ofLEByteArray data
      return toDtype.byteArrayOfIntOverflow (saturatingIntOfFloat32 toDtype f)
    | .float64, .uint8 | .float64, .uint16 | .float64, .uint32 | .float64, .uint64 => do
      let f <- Float.ofLEByteArray data
      return toDtype.byteArrayOfNatOverflow (saturatingNatOfFloat64 toDtype f)
    | .float64, .int8 | .float64, .int16 | .float64, .int32 | .float64, .int64 => do
      let f <- Float.ofLEByteArray data
      return toDtype.byteArrayOfIntOverflow (saturatingIntOfFloat64 toDtype f)
    | .float32, .float64 => do
      let f <- Float32.ofLEByteArray data
      return toLEByteArray f.toFloat
    | .float64, .float32 => do
      let f <- Float.ofLEByteArray data
      return toLEByteArray f.toFloat32
    -- fp16/bf16 to unsigned integers
    | .float16, .uint8 | .float16, .uint16 | .float16, .uint32 | .float16, .uint64
    | .bfloat16, .uint8 | .bfloat16, .uint16 | .bfloat16, .uint32 | .bfloat16, .uint64 => do
      let f <- decodeFloat16OrBFloat16 fromDtype data
      return toDtype.byteArrayOfNatOverflow (saturatingNatOfFloat32 toDtype f)
    -- fp16/bf16 to signed integers
    | .float16, .int8 | .float16, .int16 | .float16, .int32 | .float16, .int64
    | .bfloat16, .int8 | .bfloat16, .int16 | .bfloat16, .int32 | .bfloat16, .int64 => do
      let f <- decodeFloat16OrBFloat16 fromDtype data
      return toDtype.byteArrayOfIntOverflow (saturatingIntOfFloat32 toDtype f)
    -- fp16/bf16 to float32
    | .float16, .float32 | .bfloat16, .float32 => do
      let f <- decodeFloat16OrBFloat16 fromDtype data
      return toLEByteArray f
    -- fp16/bf16 to float64
    | .float16, .float64 | .bfloat16, .float64 => do
      let f <- decodeFloat16OrBFloat16 fromDtype data
      return toLEByteArray f.toFloat
    -- float32 -> fp16/bf16
    | .float32, .float16 | .float32, .bfloat16 => do
      let f <- Float32.ofLEByteArray data
      encodeFloat16OrBFloat16 toDtype f
    -- float64 -> fp16/bf16
    | .float64, .float16 | .float64, .bfloat16 => do
      let f <- Float.ofLEByteArray data
      encodeFloat16OrBFloat16 toDtype f.toFloat32
    -- fp16 <-> bf16
    | .float16, .bfloat16 | .bfloat16, .float16 => do
      let f <- decodeFloat16OrBFloat16 fromDtype data
      encodeFloat16OrBFloat16 toDtype f

    -- float8_e4m3 to unsigned integers
    | .float8_e4m3, .uint8 | .float8_e4m3, .uint16 | .float8_e4m3, .uint32 | .float8_e4m3, .uint64 => do
      let f <- decodeFloat8E4M3 data
      return toDtype.byteArrayOfNatOverflow (saturatingNatOfFloat32 toDtype f)
    -- float8_e4m3 to signed integers
    | .float8_e4m3, .int8 | .float8_e4m3, .int16 | .float8_e4m3, .int32 | .float8_e4m3, .int64 => do
      let f <- decodeFloat8E4M3 data
      return toDtype.byteArrayOfIntOverflow (saturatingIntOfFloat32 toDtype f)
    -- float8_e4m3 to float32
    | .float8_e4m3, .float32 => do
      let f <- decodeFloat8E4M3 data
      return toLEByteArray f
    -- float8_e4m3 to float64
    | .float8_e4m3, .float64 => do
      let f <- decodeFloat8E4M3 data
      return toLEByteArray f.toFloat
    -- float8_e4m3 to fp16/bf16
    | .float8_e4m3, .float16 | .float8_e4m3, .bfloat16 => do
      let f <- decodeFloat8E4M3 data
      encodeFloat16OrBFloat16 toDtype f
    -- float32 -> float8_e4m3
    | .float32, .float8_e4m3 => do
      let f <- Float32.ofLEByteArray data
      return encodeFloat8E4M3 f
    -- float64 -> float8_e4m3: rounds twice via fp32. Can disagree with ml_dtypes at the overflow edge (eg: 464.00000000000006)
    | .float64, .float8_e4m3 => do
      let f <- Float.ofLEByteArray data
      return encodeFloat8E4M3 f.toFloat32
    -- fp16/bf16 -> float8_e4m3
    | .float16, .float8_e4m3 | .bfloat16, .float8_e4m3 => do
      let f <- decodeFloat16OrBFloat16 fromDtype data
      return encodeFloat8E4M3 f

    -- float8_e5m2 to unsigned integers
    | .float8_e5m2, .uint8
    | .float8_e5m2, .uint16
    | .float8_e5m2, .uint32
    | .float8_e5m2, .uint64 => do
      let f <- decodeFloat8E5M2 data
      return toDtype.byteArrayOfNatOverflow (saturatingNatOfFloat32 toDtype f)
    -- float8_e5m2 to signed integers
    | .float8_e5m2, .int8
    | .float8_e5m2, .int16
    | .float8_e5m2, .int32
    | .float8_e5m2, .int64 => do
      let f <- decodeFloat8E5M2 data
      return toDtype.byteArrayOfIntOverflow (saturatingIntOfFloat32 toDtype f)
    -- float8_e5m2 to float32
    | .float8_e5m2, .float32 => do
      let f <- decodeFloat8E5M2 data
      return toLEByteArray f
    -- float8_e5m2 to float64
    | .float8_e5m2, .float64 => do
      let f <- decodeFloat8E5M2 data
      return toLEByteArray f.toFloat
    -- float8_e5m2 to fp16/bf16
    | .float8_e5m2, .float16
    | .float8_e5m2, .bfloat16 => do
      let f <- decodeFloat8E5M2 data
      encodeFloat16OrBFloat16 toDtype f
    -- float8_e5m2 to float8_e4m3
    | .float8_e5m2, .float8_e4m3 => do
      let f <- decodeFloat8E5M2 data
      return encodeFloat8E4M3 f
    -- float32 -> float8_e5m2
    | .float32, .float8_e5m2 => do
      let f <- Float32.ofLEByteArray data
      return encodeFloat8E5M2 f
    -- float64 -> float8_e5m2
    | .float64, .float8_e5m2 => do
      let f <- Float.ofLEByteArray data
      return encodeFloat8E5M2 f.toFloat32
    -- fp16/bf16 -> float8_e5m2
    | .float16, .float8_e5m2
    | .bfloat16, .float8_e5m2 => do
      let f <- decodeFloat16OrBFloat16 fromDtype data
      return encodeFloat8E5M2 f
    -- float8_e4m3 -> float8_e5m2
    | .float8_e4m3, .float8_e5m2 => do
      let f <- decodeFloat8E4M3 data
      return encodeFloat8E5M2 f


    | .float8_e5m2, .float8_e5m2 | .float8_e4m3, .float8_e4m3 | .float16, .float16 | .bfloat16, .bfloat16 | .float32, .float32 | .float64, .float64 => impossible


def isZero! (dtype : Dtype) (x : ByteArray) : Bool := get! $ dtype.isZero x

def nonZero (dtype : Dtype) (x : ByteArray) : Err Bool := (dtype.isZero x).map not

def nonZero! (dtype : Dtype) (x : ByteArray) : Bool := get! $ dtype.nonZero x

-- In one sense it's annoying to have an abbreviation, but it also saves some
-- cognitive effort required to translate between numbers and bools.
def logicalNot : Dtype -> ByteArray -> Err Bool := isZero

#guard Dtype.uint64.isZero! $ toLEByteArray (0 : UInt64)
#guard Dtype.uint64.isZero! $ toLEByteArray (-0 : Int64)
#guard (toLEByteArray (0.0 : Float32)).data.all fun x => x == 0
#guard !(toLEByteArray (-0.0 : Float32)).data.all fun x => x == 0
#guard (toLEByteArray (0.0 : Float)).data.all fun x => x == 0
#guard !(toLEByteArray (-0.0 : Float)).data.all fun x => x == 0
#guard Dtype.float32.isZero! $ toLEByteArray (-0.0 : Float32)
#guard Dtype.float64.isZero! $ toLEByteArray (-0.0 : Float)

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

#guard logicalAnd! Dtype.uint8 (toLEByteArray (1:UInt8)) Dtype.float64 (toLEByteArray (5:Float))
#guard !logicalAnd! Dtype.uint8 (toLEByteArray (1:UInt8)) Dtype.float64 (toLEByteArray (-0:Float))
#guard logicalOr! Dtype.uint8 (toLEByteArray (1:UInt8)) Dtype.float64 (toLEByteArray (-0:Float))
#guard logicalOr! Dtype.uint8 (toLEByteArray (0:UInt8)) Dtype.float64 (toLEByteArray (-0.1:Float))
#guard !logicalOr! Dtype.uint8 (toLEByteArray (0:UInt8)) Dtype.float64 (toLEByteArray (-0.0:Float))
#guard !logicalXor! Dtype.uint8 (toLEByteArray (0:UInt8)) Dtype.float64 (toLEByteArray (-0.0:Float))

/-
When you call real functions on int arrays, for example, NumPy converts the array to some float
type before calling the function. This follows some rules, which we approximate here, given we don't
have all the types available in NumPy and we may have types like bfloat that aren't
in NumPy out of the box (but are available via ml_dtypes).
-/
-- Float types remain unchanged
-- intx and uintx map to the smallest float that holds their range
-- isFloat check covers fp16, bf16, fp32, and fp64
-- note : bf16 will not be changed to fp32 since it will be caught in the if block
def floatVariant (dtype : Dtype) : Dtype :=
  if dtype.isFloat then dtype else if dtype.itemsize <= 2 then .float32 else .float64

private def liftFloatUnop (f32 : Float32 -> Err Float32) (f64 : Float -> Err Float)
                          (dtype : Dtype) (data : ByteArray) : Err ByteArray := do
  if data.size != dtype.itemsize then throw "incorrect byte count" else
  match dtype with
  | .float8_e5m2 => do
    let f <- decodeFloat8E5M2 data
    let x <- f32 f
    return encodeFloat8E5M2 x
  | .float8_e4m3 => do
    let f <- decodeFloat8E4M3 data
    let x <- f32 f
    return encodeFloat8E4M3 x
  | .float16 | .bfloat16 => do
    let f <- decodeFloat16OrBFloat16 dtype data
    let x <- f32 f
    encodeFloat16OrBFloat16 dtype x
  | .float32 => do
    let f <- Float32.ofLEByteArray data
    let x <- f32 f
    return toLEByteArray x
  | .float64 => do
    let f <- Float.ofLEByteArray data
    let x <- f64 f
    return toLEByteArray x
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
| .float32 | .float64 | .bfloat16 | .float16 | .float8_e4m3 | .float8_e5m2 => throw "shifts not supported at float type"
| .bool => throw "In NumPy, bool shifts are cast to int64. This seems arbitrary so please cast (e.g. with astype) before you shift."
| .uint64 | .int64 | .uint32 | .int32 | .uint16 | .int16 | .uint8 | .int8 =>
  let k := dtype.itemsize
  if bits.size != k then throw "dtype size mismatch" else
  let n64 := bits.toNat.toUInt64
  let shift64 := shiftAmount.toNat.toUInt64
  let n64 := f n64 shift64
  return (toLEByteArray n64).take k

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
  let arr := toLEByteArray n
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
  let arr := toLEByteArray n
  let sin := typ.sin! arr
  let cos := typ.cos! arr
  let div := typ.div! sin cos
  let tan := typ.tan! arr
  closeEnough32! div tan

-- log (e ^ x) = x is bitwise accurate at float64, at least for the numbers we try
#guard
  let typ := Dtype.float64
  let n : Float := 1.1
  let arr := toLEByteArray n
  let exp := typ.exp! arr
  let log := typ.log! exp
  closeEnough64! log arr

-- log (e ^ x) = x is also bitwise accurate at float32, at least for the numbers we try
#guard
  let typ := Dtype.float32
  let n : Float32 := 1.1
  let arr := toLEByteArray n
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
info: Unable to find a counter-example
---
warning: declaration uses 'sorry'
-/
#guard_msgs in
example (fromDtype toDtype : Dtype) (n : Nat) :
  canCastLosslessIntRoundTrip fromDtype 0 toDtype &&
  canCastLosslessIntRoundTrip fromDtype 1 toDtype
  := by plausible

/--
info: Unable to find a counter-example
---
warning: declaration uses 'sorry'
-/
#guard_msgs in
-- One dtype should always go back and forth
-- skip values outside dtypes range since they cannot be encoded in the first place.
example (dtype : Dtype) (n : Nat) :
  if n > dtype.maxSafeNat.getD n then true else canCastLosslessIntRoundTrip dtype n dtype := by plausible

/--
info: Unable to find a counter-example
---
warning: declaration uses 'sorry'
-/
#guard_msgs in
-- Lossless translations should be OK
example (fromDtype toDtype : Dtype) (n : Nat) :
-- only try values within fromDtye's representable range
-- fp16 max n = 2^11, fp32 max n = 2^24, fp64 = 2^53, uint8 = 255, int8 = 127, etc
  if lossless fromDtype toDtype then
    if n > fromDtype.maxSafeNat.getD n then true else canCastLosslessIntRoundTrip fromDtype n toDtype
  else true := by plausible


-- fp16 addition is commutative (a + b == b + a)
/--
info: Unable to find a counter-example
---
warning: declaration uses 'sorry'
-/
#guard_msgs in
example (a b : UInt16) :
  let xa := toLEByteArray a
  let xb := toLEByteArray b
  Dtype.add .float16 xa xb == Dtype.add .float16 xb xa := by plausible


-- Property: fp16 adding zero is identity (a + 0 == a) for non-NaN values
/--
info: Unable to find a counter-example
---
warning: declaration uses 'sorry'
-/
#guard_msgs in
example (a : UInt16) :
  let xa := toLEByteArray a
  let zero := toLEByteArray (0 : UInt16)
  let f := a.toFloat32FromFloat16
   -- f != f is true only for NaN; check identity for everything else
   -- if value is +-0 skip byte equality check since addition is identity upto sign
  f != f ∨ f == 0 ∨ Dtype.add .float16 xa zero == .ok xa := by plausible


-- Property: bf16 addition is commutative
/--
info: Unable to find a counter-example
---
warning: declaration uses 'sorry'
-/
#guard_msgs in
example (a b : UInt16) :
  let xa := toLEByteArray a
  let xb := toLEByteArray b
  Dtype.add .bfloat16 xa xb == Dtype.add .bfloat16 xb xa := by plausible

-- Property: e4m3 addition is commutative (a + b == b + a)
/--
info: Unable to find a counter-example
---
warning: declaration uses 'sorry'
-/
#guard_msgs in
example (a b : UInt8) :
  let xa := toLEByteArray a
  let xb := toLEByteArray b
  Dtype.add .float8_e4m3 xa xb == Dtype.add .float8_e4m3 xb xa := by plausible

-- PBT for join commutativity
-- Since joinOrdered requires both arguments to be listed for same size types (the swap guard is triggered only when sizes differ)
-- This PBT catches any missing direction that would silently return none instead of promoting
/--
info: Unable to find a counter-example
---
warning: declaration uses 'sorry'
-/
#guard_msgs in
example (a b : Dtype) : Dtype.join a b == Dtype.join b a := by plausible


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
