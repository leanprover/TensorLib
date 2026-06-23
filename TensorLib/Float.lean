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

namespace TensorLib

/-
The largest Nat such that it and every smaller Nat can be represented exactly in a 64-bit IEEE-754 float
https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number/MAX_SAFE_INTEGER

The number 2^(mantissa size).

https://en.wikipedia.org/wiki/Floating-point_arithmetic
https://en.wikipedia.org/wiki/Double-precision_floating-point_format
-/
private def float32MantissaBits : Nat := 23
private def float64MantissaBits : Nat := 52
private def float16MantissaBits : Nat := 10

-- Add 1 to the mantissa length because of the implicit leading 1
def maxSafeNatForFloat32 : Nat := Nat.pow 2 (float32MantissaBits + 1)
def maxSafeNatForFloat64 : Nat := Nat.pow 2 (float64MantissaBits + 1)
def maxSafeNatForFloat16 : Nat := Nat.pow 2 (float16MantissaBits + 1)

def _root_.Float32.minValue : Float32 := Float32.ofBits 0xFF7FFFFF
def _root_.Float32.maxValue : Float32 := Float32.ofBits 0x7F7FFFFF
def _root_.Float.minValue : Float := Float.ofBits 0xFFEFFFFFFFFFFFFF
def _root_.Float.maxValue : Float := Float.ofBits 0x7FEFFFFFFFFFFFFF

def _root_.Int.toFloat32 (n : Int) : Float32 := Float32.ofInt n
def _root_.Int.toFloat64 (n : Int) : Float := Float.ofInt n

instance : ToLEByteArray Float32 where
  toLEByteArray f := toLEByteArray f.toBits

instance : ToBEByteArray Float32 where
  toBEByteArray f := toBEByteArray f.toBits

instance : ToLEByteArray Float where
  toLEByteArray f := toLEByteArray f.toBits

instance : ToBEByteArray Float where
  toBEByteArray f := toBEByteArray f.toBits

def _root_.Float32.ofLEByteArray (arr : ByteArray) : Err Float32 := do
  return Float32.ofBits (<- arr.toUInt32LE)

def _root_.Float32.ofBEByteArray (arr : ByteArray) : Err Float32 := do
  return Float32.ofBits (<- arr.toUInt32BE)

def _root_.Float32.ofLEByteArray! (arr : ByteArray) : Float32 := get! $ Float32.ofLEByteArray arr
def _root_.Float32.ofBEByteArray! (arr : ByteArray) : Float32 := get! $ Float32.ofBEByteArray arr

def _root_.Float.ofLEByteArray (arr : ByteArray) : Err Float := do
  return Float.ofBits (<- arr.toUInt64LE)

def _root_.Float.ofBEByteArray (arr : ByteArray) : Err Float := do
  return Float.ofBits (<- arr.toUInt64BE)

def _root_.Float.ofLEByteArray! (arr : ByteArray) : Float := get! $ Float.ofLEByteArray arr
def _root_.Float.ofBEByteArray! (arr : ByteArray) : Float := get! $ Float.ofBEByteArray arr

def _root_.Float32.toNat (f : Float32) : Nat := f.toUInt64.toNat

def _root_.Float32.toInt (f : Float32) : Int :=
  let neg := f <= 0
  let f := if neg then -f else f
  let n := Int.ofNat f.toUInt64.toNat
  if neg then -n else n

def _root_.Float32.quietNaN : Float32 := Float32.ofBits 0x7FC00000

#guard Float32.quietNaN.toInt == 0

def _root_.Float.toNat (f : Float) : Nat := f.toUInt64.toNat

def _root_.Float.toInt (f : Float) : Int :=
  let neg := f <= 0
  let f := if neg then -f else f
  let n := Int.ofNat f.toUInt64.toNat
  if neg then -n else n

def _root_.Float.quietNaN : Float := Float.ofBits 0x7FF8000000000000

-- Convert a float32 to Float16 as UInt16
def _root_.Float32.toFloat16Bits (f : Float32) : UInt16 :=
  let bits := f.toBits
  let sign := (bits >>> 31) &&& 1
  let exp := (bits >>> 23) &&& 0xFF
  let mant := bits &&& 0x7FFFFF
  let sign16 := sign.toUInt16 <<< 15
  if exp == 0 then sign16 -- 0 or subnormal float32 is 0 in float16
  else if exp == 0xFF then sign16 ||| ((0x1F : UInt16) <<< 10) ||| (mant >>> 13).toUInt16 -- if Inf or NaN. 0x1F
  else
    let newExp : Int := exp.toNat - 127 + 15 -- rebias from float32 to float16 bias
    if newExp >= 0x1F then sign16 ||| ((0x1F : UInt16) <<< 10) -- overflow -> inf
    else if newExp <= 0 then sign16 -- underflow -> 0
    else sign16 ||| (newExp.toNat.toUInt16 <<< 10) ||| (mant >>> 13).toUInt16


-- Convert float16 bits (stored as UInt) to float32.
-- Extract sign, exp, and mantissa from 16 bit representation
-- Check for zero, inf, NaN
-- If number is none of the above then adjust bias (add 127 - 15)
-- left shift mantissa 13
-- pack it into 32 bit layout
def _root_.UInt16.toFloat32FromFloat16 (bits : UInt16) : Float32 :=
  let sign := (bits >>> 15) &&& 1  -- sign bit; positive = 0, negative = 1
  let exp := (bits >>> 10) &&& 0x1F -- extract bits 14..10
  let mant := bits &&& 0x3FF -- mantissa bits
  if exp ==  0 && mant == 0 then
    if sign == 1 then Float32.ofBits 0x80000000 -- -0
    else Float32.ofBits 0  -- +0
  else if exp == 0x1FF then -- exp is all 1's so this is either +-inf or NaN
    if mant == 0 then Float32.ofBits (sign.toUInt32 <<< 32 ||| 0x7F800000) -- +- inf
    else Float32.ofBits 0x7FC00000 -- Nan
  else
    -- Rebias the exponent from float16 (15) bias to float32 (127)
    -- shift mant from 10 bits to 23 bits so pad with 13 0's
    let newExp := exp.toUInt32 - 15 + 127
    Float32.ofBits (sign.toUInt32 <<< 31 ||| newExp <<< 23 ||| mant.toUInt32 <<< 13)

#guard Float.quietNaN.toInt == 0

section Test

#guard (
  let n : UInt64 := 0x3FFAB851EB851EB8
  do
    let arr := toLEByteArray n
    let n' <- arr.toUInt64LE
    return n == n') == .ok true

-- Tests showing that the last number we can represent is maxSafeNatForFloat{32,64}
#guard
  let n := maxSafeNatForFloat32 + 1
  let f := n.toFloat32
  f.toFloat.toNat == n - 1

#guard
  let n := maxSafeNatForFloat64 + 1
  let f := n.toFloat
  f.toNat == n - 1

end Test

end TensorLib
