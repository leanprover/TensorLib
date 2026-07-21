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
private def bfloat16MantissaBits : Nat := 7
private def float8e4m3MantissaBits : Nat := 3
private def float8e5m2MantissaBits : Nat := 2

-- Add 1 to the mantissa length because of the implicit leading 1
def maxSafeNatForFloat32 : Nat := Nat.pow 2 (float32MantissaBits + 1)
def maxSafeNatForFloat64 : Nat := Nat.pow 2 (float64MantissaBits + 1)
def maxSafeNatForFloat16 : Nat := Nat.pow 2 (float16MantissaBits + 1)
def maxSafeNatForBFloat16 : Nat := Nat.pow 2 (bfloat16MantissaBits + 1)
def maxSafeNatForFloat8e4m3 : Nat := Nat.pow 2 (float8e4m3MantissaBits + 1)
def maxSafeNatForFloat8e5m2 : Nat := Nat.pow 2 (float8e5m2MantissaBits + 1)

def _root_.Float32.minValue : Float32 := Float32.ofBits 0xFF7FFFFF
def _root_.Float32.maxValue : Float32 := Float32.ofBits 0x7F7FFFFF
def _root_.Float.minValue : Float := Float.ofBits 0xFFEFFFFFFFFFFFFF
def _root_.Float.maxValue : Float := Float.ofBits 0x7FEFFFFFFFFFFFFF

def _root_.Float32.isPosInf (f : Float32) : Bool := f == Float32.ofBits 0x7F800000
def _root_.Float32.isNegInf (f : Float32) : Bool := f == Float32.ofBits 0xFF800000


def _root_.Float.isPosInf (f : Float) : Bool := f == Float.ofBits 0x7FF0000000000000
def _root_.Float.isNegInf (f : Float) : Bool := f == Float.ofBits 0xFFF0000000000000

def _root_.Int.toFloat32 (n : Int) : Float32 := Float32.ofInt n
def _root_.Int.toFloat64 (n : Int) : Float := Float.ofInt n

-- IEEE 754 equality on infinities
#guard (Float32.ofBits 0x7F800000).isPosInf
#guard (Float32.ofBits 0xFF800000).isNegInf

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

-- NaN and negatives (including -inf) → 0; +inf → UINT64_MAX
-- Truncation in byteArrayOfNatOverflow gives correct UINT_MAX per target size
-- (e.g. UINT64_MAX mod 256 = 255 for uint8)
def _root_.Float32.toNat (f : Float32) : Nat :=
  if f.isNaN then 0          -- NaN -> 0
  else if f <= 0 then 0     -- -inf and negatives -> 0
  else f.toUInt64.toNat     -- +inf -> UINT64_MAX

-- Returns INT64_MAX/MIN for ±inf, 0 for NaN.
-- Per-dtype saturation (e.g. +inf → INT8_MAX for int8) is handled by
-- saturatingIntOfFloat32/saturatingIntOfFloat64 in the cast paths.
def _root_.Float32.toInt (f : Float32) : Int :=
  if f.isNaN then 0
  else if f.isNegInf then -0x8000000000000000 -- -inf -> INT64_MIN
  else if f.isPosInf then 0x7FFFFFFFFFFFFFFF -- +inf -> INT64_MAX
  else
    let neg := f <= 0
    let f := if neg then -f else f
    let n := Int.ofNat f.toUInt64.toNat
    if neg then -n else n

def _root_.Float32.quietNaN : Float32 := Float32.ofBits 0x7FC00000

#guard Float32.quietNaN.toInt == 0

-- NaN and negatives (including -inf) → 0; +inf → UINT64_MAX
def _root_.Float.toNat (f : Float) : Nat :=
  if f != f then 0
  else if f <= 0 then 0
  else f.toUInt64.toNat

-- Same inf/NaN handling as Float32.toInt (see TODO comment above)
def _root_.Float.toInt (f : Float) : Int :=
  if f != f then 0   -- NaN -> 0
  else if f == Float.ofBits 0xFFF0000000000000 then -0x8000000000000000  -- -inf -> INT64_MIN
  else if f == Float.ofBits 0x7FF0000000000000 then 0x7FFFFFFFFFFFFFFF   -- +inf -> INT64_MAX
  else
    let neg := f <= 0
    let f := if neg then -f else f
    let n := Int.ofNat f.toUInt64.toNat
    if neg then -n else n

def _root_.Float.quietNaN : Float := Float.ofBits 0x7FF8000000000000

-- Convert a float32 to Float16 as UInt16
-- NaN sign is not preserved because Lean's fp32 normalizes NaN to +NaN irrespective of input sign which diverges from ml_dtypes (ml_dtypes preserves sign of NaN)
def _root_.Float32.toFloat16Bits (f : Float32) : UInt16 :=
  let bits := f.toBits
  let sign := (bits >>> 31) &&& 1
  let exp := (bits >>> 23) &&& 0xFF
  let mant := bits &&& 0x7FFFFF
  let sign16 := sign.toUInt16 <<< 15
  if exp == 0 then sign16 -- 0 or subnormal float32 is 0 in float16
  else if exp == 0xFF then -- if Inf or NaN. 0x1F
    if mant == 0 then
      -- infinity
      sign16 ||| ((0x1F : UInt16) <<< 10)
    else
      -- NaN : make sure mantissa is nonzero in fp16 to preserve NaN-ness
      let truncMant := (mant >>> 13).toUInt16
      let nanMant := if truncMant == 0 then (0x0200 : UInt16) else truncMant -- set quiet Nan bit if payload lost
      sign16 ||| ((0x1f : UInt16) <<< 10) ||| nanMant
  else
    let newExp : Int := exp.toNat - 127 + 15 -- rebias from float32 to float16 bias
    if newExp >= 0x1F then sign16 ||| ((0x1F : UInt16) <<< 10) -- overflow -> inf
    else if newExp <= 0 then -- underflow for subnormal in fp16
        -- Work with full fp32 mantissa + implicit leading 1 bit
        -- Total shift combines fp32 to fp16 mantissa narrowing (13) + subnormal adjustment (1-newExp)
        let totalShift := (14 - newExp).toNat
        -- If shift >= 25, all 24 significant bits are gone -> value should be ±zero
        if totalShift >= 25 then sign16
        else
          let fullMant := mant ||| 0x800000  -- add implicit leading 1 at bit 23
          -- Round-to-nearest-even on discarded bits
          let roundBit := if totalShift > 0 then (fullMant >>> (totalShift.toUInt32 - 1)) &&& 1 else 0
          let stickyMask := if totalShift > 1 then (1 <<< (totalShift.toUInt32 - 1)) - 1 else 0
          let stickyBits := fullMant &&& stickyMask
          let shifted := fullMant >>> totalShift.toUInt32
          let rounded := if roundBit == 1 && (stickyBits != 0 || shifted &&& 1 == 1)
            then shifted + 1
            else shifted
          sign16 ||| rounded.toUInt16
    else
      -- Round to nearest even
      let truncated := mant >>> 13
      let roundBit := (mant >>> 12) &&& 1 -- highest discarded bit
      let stickyBits := mant &&& 0xFFF -- the 12 leftover bits
      -- Round up if roundBit is 1 and either stickyBits nonzero or last kept bit is odd
      let rounded := if roundBit == 1 && (stickyBits != 0 || truncated &&& 1 == 1) then truncated + 1 else truncated
      -- Mantissa overflow ronding up 0x3FF to 0x400 means we need to bump the exp
      let (finalExp, finalMant) := if rounded > 0x3FF
        then (newExp.toNat.toUInt16 + 1, (0 : UInt16))
        else (newExp.toNat.toUInt16, rounded.toUInt16)
      -- pack it all into fp16 using rounded values instead of truncating
      sign16 ||| (finalExp <<< 10) ||| finalMant

-- Convert float32 to BFloat16
-- bf16 is the top 16 bits of fp32 with round to nearest even on the discarded bits
-- no exponent rebiasing needed since it's equal to fp32's
-- Subnormals are taken care of since bf16 sub = fp32 sub, same exponent range
-- Sign is preserved  in normal values
-- NaN sign is not preserved because Lean's fp32 type normalizes NaN to +NaN irrespective of input sign.
-- This diverges from ml_dtypes which preserves sign of NaN.
def _root_.Float32.toBFloat16Bits (f : Float32) : UInt16 :=
  let bits := f.toBits
  let top := bits >>> 16 -- top 16 bits
  -- bit 15 determines if we are >= 0.5 ULP
  let roundBit := (bits >>> 15) &&& 1
  -- If any remainder bits are non zero we are > 0.5 ULP
  let stickyBits := bits &&& 0x7FFF
  -- Round to nearest even
  -- round up if discarded part >= 0.5 ULP and last kept bit is odd
  let rounded := if roundBit == 1 && (stickyBits != 0 || top &&& 1 == 1) then top + 1 else top
  -- truncate to bf16 representation
  rounded.toUInt16


-- Convert float16 bits (stored as UInt) to float32.
-- Extract sign, exp, and mantissa from 16 bit representation
-- Check for zero, inf, NaN, or subnormal
-- If number is none of the above then adjust bias (add 127 - 15)
-- left shift mantissa 13
-- pack it into 32 bit layout
def _root_.UInt16.toFloat32FromFloat16 (bits : UInt16) : Float32 :=
  let sign := (bits >>> 15) &&& 1  -- sign bit; positive = 0, negative = 1
  let exp := (bits >>> 10) &&& 0x1F -- extract bits 14..10
  let mant := bits &&& 0x3FF -- mantissa bits
  if exp ==  0 then
    -- Number can be either +-0 or subnormal
    if mant == 0 then
      -- All bits 0 in mantissa -> +-0
      if sign == 1 then Float32.ofBits 0x80000000 else Float32.ofBits 0 -- +0
    else
      -- subnormal fp16 : exp is 0 but mant is != 0
      -- value = (-1)^sign x mant x 2 ^ (-24)
      let f := Float32.ofNat mant.toNat -- mant <= 1023 fits in fp32
      let scale := Float32.ofBits 0x33800000 -- 2 ^ (-24) in fp32
      let result := f * scale -- shift the exponent, no precision lost
      -- Apply sign bit
      if sign == 1 then Float32.ofBits (result.toBits ||| 0x80000000)
      else result
  else if exp == 0x1F then -- exp is all 1's so this is either +-inf or NaN
    if mant == 0 then Float32.ofBits (sign.toUInt32 <<< 31 ||| 0x7F800000) -- +- inf
    -- puts the original sign bit at position 31, so negative fp16 NaN stays negative in fp32.
     else Float32.ofBits (sign.toUInt32 <<< 31 ||| 0x7FC00000) -- NaN
  else
    -- Rebias the exponent from float16 (15) bias to float32 (127)
    -- shift mant from 10 bits to 23 bits so pad with 13 0's
    let newExp := exp.toUInt32 - 15 + 127
    Float32.ofBits (sign.toUInt32 <<< 31 ||| newExp <<< 23 ||| mant.toUInt32 <<< 13)


#guard Float.quietNaN.toInt == 0

-- Convert bfloat16 bits (UInt16) to fp32
-- bf16 is the top 16 bits of fp32 so shift left by 16
-- subnormals: padded bit pattern is valid fp32 subnorm
-- sign bit: preserved
-- inf/NaN is handled automatically since all 1's in exp stays valid after padding
def _root_.UInt16.toFloat32FromBFloat16 (bits : UInt16) : Float32 :=
  Float32.ofBits (bits.toUInt32 <<< 16)

-- Convert float8_e4m3fn bits (stored as UInt8) to Float32.
-- E4M3FN format: 1 sign bit + 4 exponent bits + 3 mantissa bits, bias = 7
-- Special values: No inf. Exponent all 1's (0xF) with mantissa 0x7 = NaN; mantissa 0..6 are normal (up to 448).
-- 0x7F = +NaN, 0xFF = -NaN (only two NaN encodings since mant must be 0x7)
-- Max value: ±448 (bits 0x7E / 0xFE)
-- Subnormal: exp=0, mant≠0, value = (-1)^sign × mant × 2^(1 - bias - mantissa_bits) = mant × 2^(-9)
-- Smallest subnormal: 2^(-9) = 0.001953125
-- Smallest normal: 2^(-6) = 0.015625
def _root_.UInt8.toFloat32FromFloat8E4M3 (bits : UInt8) : Float32 :=
  let sign := (bits >>> 7) &&& 1        -- bit 7: sign
  let exp := (bits >>> 3) &&& 0xF       -- bits 6..3: exponent (4 bits)
  let mant := bits &&& 0x7              -- bits 2..0: mantissa (3 bits)
  let sign32 := sign.toUInt32 <<< 31    -- sign bit in fp32 position
  if exp == 0 then
    if mant == 0 then
      -- +- 0
      Float32.ofBits sign32
    else
      -- Subnormal: no implicit leading 1
      -- value = (-1)^sign × mant × 2^(1 - 7 - 3) = mant × 2^(-9)
      let f := Float32.ofNat mant.toNat
      let scale := Float32.ofBits 0x3B000000  -- 2^(-9) in fp32
      let result := f * scale
      if sign == 1 then Float32.ofBits (result.toBits ||| 0x80000000) else result
  else if exp == 0xF && mant == 0x7 then
    -- Exponent all 1s: NaN (e4m3fn has NO infinity, only NaN)
    -- We map all NaN bit patterns to fp32 quiet NaN.
    -- NaN sign is not preserved due to leans fp32 NaN normalization, diverging from ml_dtypes.
    Float32.ofBits (sign32 ||| 0x7FC00000)
  else
    -- Normal number: rebias exponent from e4m3 (bias=7) to fp32 (bias=127)
    -- Shift mantissa from 3 bits to 23 bits (left-pad with 20 zeros)
    let newExp := exp.toUInt32 - 7 + 127
    Float32.ofBits (sign32 ||| newExp <<< 23 ||| mant.toUInt32 <<< 20)

-- Convert Fp32 to float8_e4m3 bits (UInt8).
-- E4M3: 1 sign + 4 exponent + 3 mantissa, bias = 7
-- Uses round-to-nearest-even on discarded mantissa bits.
-- Overflow (including +-inf) maps to NaN (0x7F / 0xFF) per ml_dtypes behavior.
-- NaN sign is not preserved because lean's fp32 normalizes NaN to +NaN irrespective of input sign which diverges from ml_dtypes.
def _root_.Float32.toFloat8E4M3Bits (f : Float32) : UInt8 :=
  let bits := f.toBits
  let sign := (bits >>> 31) &&& 1
  let exp := (bits >>> 23) &&& 0xFF
  let mant := bits &&& 0x7FFFFF
  let sign8 := sign.toUInt8 <<< 7
  if exp == 0xFF then
    -- Input is +-inf or NaN -> e4m3 NaN
    sign8 ||| 0x7F
  else if exp == 0 then
    -- Input is zero or fp32 subnormal -> too small for e4m3 so flush to zero
    sign8
  else
    -- Normal fp32 value. Rebias exponent from fp32 (127) to e4m3 (7).
    let realExp : Int := exp.toNat - 127
    -- e4m3 normal range: realExp in [-6, 8] (exp field 1..15, but 15+mant=7 is NaN)
    -- e4m3 subnormal range: realExp = -6 with reduced mantissa
    -- Full mantissa with implicit leading 1 (24 bits)
    let fullMant := mant ||| 0x800000
    if realExp > 8 then
      -- Overflow -> NaN (e4m3fn has no inf, overflow = NaN per ml_dtypes)
      sign8 ||| 0x7F
    else if realExp > 7 then
      -- realExp == 8: only valid if mantissa rounds to <= 0b110 (i.e. value <= 448)
      -- e4m3 at exp=14 (realExp=7), mant=0b111 -> (1+7/8)*2^7 = 240, but exp=15 with mant 0..6 goes up to 448.
      -- Actually exp=15 is valid for mant 0..6 (mant=7 is NaN). realExp = 15-7 = 8.
      -- value = (1 + mant/8) * 2^8. Max = (1+6/8)*256 = 448.
      -- So realExp=8, truncated mant must be <= 6.
      -- Shift: we need 3 mantissa bits from 23 fp32 mantissa bits -> shift right by 20
      let truncated := fullMant >>> 20
      let roundBit := (fullMant >>> 19) &&& 1
      let stickyBits := fullMant &&& 0x7FFFF
      let rounded := if roundBit == 1 && (stickyBits != 0 || truncated &&& 1 == 1)
        then truncated + 1 else truncated
      -- rounded includes implicit 1 at bit 3, so subtract it: rounded - 8
      let mantBits := rounded - 8
      if mantBits > 6 then
        -- Rounded past max → NaN
        sign8 ||| 0x7F
      else
        sign8 ||| (15 : UInt8) <<< 3 ||| mantBits.toUInt8
    else if realExp >= -6 then
      -- Normal e4m3 range: realExp in [-6, 7]
      -- e4m3 exponent field = realExp + 7 (so 1..14)
      let e4m3Exp := (realExp + 7).toNat
      -- Truncate fp32 mantissa (23 bits) to 3 bits: shift right by 20
      -- (we don't need the implicit 1 since it's implicit in e4m3 too)
      let truncated := mant >>> 20
      let roundBit := (mant >>> 19) &&& 1
      let stickyBits := mant &&& 0x7FFFF
      let rounded := if roundBit == 1 && (stickyBits != 0 || truncated &&& 1 == 1)
        then truncated + 1 else truncated
      -- If rounding overflows mantissa (0b1000), bump exponent
      -- e4m3Exp is [1, 14] here so e4m3Exp + 1 is always in [2, 15] and never overflows.
      let (finalExp, finalMant) := if rounded > 0x7 then (e4m3Exp + 1, (0 : UInt32)) else (e4m3Exp, rounded)
      -- NaN is unreachable here. Mant carry forces finalMant = 0
      sign8 ||| (finalExp.toUInt8 <<< 3) ||| finalMant.toUInt8
    else
      -- Subnormal in e4m3: realExp < -6
      -- value = mant * 2^(-9), where mant is 1..7
      -- totalShift: how many bits to shift fullMant to get 3-bit subnormal mantissa
      -- For subnormal at realExp = -7: shift = 20 + 1 = 21
      -- fullMant is 24 bits (bit 23 = implicit 1). We need to shift it so that
      -- the result represents mant * 2^(-9).
      -- Normal at exp=-6: value = (1 + m/8) * 2^(-6), encoded as exp=1, mant=m
      -- Subnormal: value = (m/8) * 2^(-6) = m * 2^(-9)
      -- So we need fullMant * 2^(realExp) = m * 2^(-9)
      -- m = fullMant * 2^(realExp + 9) / 2^23 = fullMant >> (23 - realExp - 9) = fullMant >> (14 - realExp)
      let totalShift := (14 - realExp).toNat
      if totalShift >= 25 then
        -- All bits shifted out → zero
        sign8
      else
        let shifted := fullMant >>> totalShift.toUInt32
        let roundBit := if totalShift > 0 then (fullMant >>> (totalShift.toUInt32 - 1)) &&& 1 else 0
        let stickyMask := if totalShift > 1 then (1 <<< (totalShift.toUInt32 - 1)) - 1 else 0
        let stickyBits := fullMant &&& stickyMask
        let rounded := if roundBit == 1 && (stickyBits != 0 || shifted &&& 1 == 1)
          then shifted + 1 else shifted
        -- If rounded up to 8, it becomes the smallest normal (exp=1, mant=0)
        if rounded >= 8 then
          sign8 ||| (1 : UInt8) <<< 3
        else
          sign8 ||| rounded.toUInt8

-- Convert float8_e5m2 bits (stored as UInt8) to Float32.
-- E5M2 format: 1 sign bit + 5 exponent bits + 2 mantissa bits, bias = 15
-- Special values: exp=0x1F, mant=0 → ±inf; exp=0x1F, mant≠0 → NaN
-- Max value: ±57344 (bits 0x7B / 0xFB)
-- Subnormal: exp=0, mant≠0, value = (-1)^sign × mant × 2^(-16)
-- Smallest subnormal: 2^(-16)
-- Smallest normal: 2^(-14)
def _root_.UInt8.toFloat32FromFloat8E5M2 (bits : UInt8) : Float32 :=
  let sign := (bits >>> 7) &&& 1        -- bit 7: sign
  let exp := (bits >>> 2) &&& 0x1F      -- bits 6..2: exponent (5 bits)
  let mant := bits &&& 0x3              -- bits 1..0: mantissa (2 bits)
  let sign32 := sign.toUInt32 <<< 31    -- sign bit in fp32 position
  if exp == 0 then
    if mant == 0 then
      -- ±zero
      Float32.ofBits sign32
    else
      -- Subnormal: no implicit leading 1
      -- value = (-1)^sign × mant × 2^(1 - 15 - 2) = mant × 2^(-16)
      let f := Float32.ofNat mant.toNat
      let scale := Float32.ofBits 0x37800000  -- 2^(-16) in fp32
      let result := f * scale
      if sign == 1 then Float32.ofBits (result.toBits ||| 0x80000000)
      else result
  else if exp == 0x1F then
    if mant == 0 then
      -- ±infinity
      Float32.ofBits (sign32 ||| 0x7F800000)
    else
      -- NaN (multiple NaN encodings: mant = 1, 2, or 3)
      -- NaN sign not preserved due to Lean's Fp32 normalization.
      Float32.ofBits (sign32 ||| 0x7FC00000)
  else
    -- Normal: rebias exponent from e5m2 (bias=15) to fp32 (bias=127)
    -- Shift mantissa from 2 bits to 23 bits (left-pad with 21 zeros)
    let newExp := exp.toUInt32 - 15 + 127
    Float32.ofBits (sign32 ||| newExp <<< 23 ||| mant.toUInt32 <<< 21)

-- e5m2 decode tests (verified against ml_dtypes)
#guard (0 : UInt8).toFloat32FromFloat8E5M2 == Float32.ofBits 0x00000000       -- +0
#guard (128 : UInt8).toFloat32FromFloat8E5M2 == Float32.ofBits 0x80000000     -- -0
#guard (60 : UInt8).toFloat32FromFloat8E5M2 == 1.0                            -- 1.0
#guard (188 : UInt8).toFloat32FromFloat8E5M2 == -1.0                          -- -1.0
#guard (64 : UInt8).toFloat32FromFloat8E5M2 == 2.0                            -- 2.0
#guard (123 : UInt8).toFloat32FromFloat8E5M2 == Float32.ofBits 0x47600000     -- max (57344)
#guard (251 : UInt8).toFloat32FromFloat8E5M2 == Float32.ofBits 0xC7600000     -- -max (-57344)
#guard (124 : UInt8).toFloat32FromFloat8E5M2 == Float32.ofBits 0x7F800000     -- +inf
#guard (252 : UInt8).toFloat32FromFloat8E5M2 == Float32.ofBits 0xFF800000     -- -inf
#guard (1 : UInt8).toFloat32FromFloat8E5M2 == Float32.ofBits 0x37800000       -- smallest subnormal
#guard (4 : UInt8).toFloat32FromFloat8E5M2 == Float32.ofBits 0x38800000       -- smallest normal
#guard (62 : UInt8).toFloat32FromFloat8E5M2 == 1.5                            -- 1.5
-- NaN: f != f
#guard (125 : UInt8).toFloat32FromFloat8E5M2 != (125 : UInt8).toFloat32FromFloat8E5M2  -- NaN
#guard (127 : UInt8).toFloat32FromFloat8E5M2 != (127 : UInt8).toFloat32FromFloat8E5M2  -- NaN


-- Convert Float32 to float8_e5m2 bits (UInt8).
-- E5M2: 1 sign + 5 exponent + 2 mantissa, bias = 15
-- Uses round-to-nearest-even on discarded mantissa bits.
-- Overflow maps to ±inf (unlike e4m3 which maps to NaN).
-- NaN sign not reliably preserved due to Lean's Float32 NaN normalization.
def _root_.Float32.toFloat8E5M2Bits (f : Float32) : UInt8 :=
  let bits := f.toBits
  let sign := (bits >>> 31) &&& 1
  let exp := (bits >>> 23) &&& 0xFF
  let mant := bits &&& 0x7FFFFF
  let sign8 := sign.toUInt8 <<< 7
  if exp == 0xFF then
    if mant == 0 then
      -- +-inf -> +-inf in e5m2
      sign8 ||| 0x7C
    else
      -- NaN → quiet NaN in e5m2
      sign8 ||| 0x7E
  else if exp == 0 then
    -- fp32 zero or subnormal is too small for e5m2, flush to zero
    sign8
  else
    -- Normal fp32 value. Rebias exponent from fp32 (127) to e5m2 (15).
    let realExp : Int := exp.toNat - 127
    let fullMant := mant ||| 0x800000
    if realExp > 15 then
      -- Overflow → ±inf
      sign8 ||| 0x7C
    else if realExp >= -14 then
      -- Normal e5m2 range: realExp in [-14, 15]
      -- e5m2 exponent field = realExp + 15 (so 1..30)
      let e5m2Exp := (realExp + 15).toNat
      -- Truncate fp32 mantissa (23 bits) to 2 bits: shift right by 21
      let truncated := mant >>> 21
      let roundBit := (mant >>> 20) &&& 1
      let stickyBits := mant &&& 0xFFFFF
      let rounded := if roundBit == 1 && (stickyBits != 0 || truncated &&& 1 == 1)
        then truncated + 1 else truncated
      -- If rounding overflows mantissa (0b100), bump exponent
      let (finalExp, finalMant) := if rounded > 0x3 then
        (e5m2Exp + 1, (0 : UInt32))
      else (e5m2Exp, rounded)
      -- If exponent overflows to 31 with mant=0, that's inf
      if finalExp >= 31 then
        sign8 ||| 0x7C
      else
        sign8 ||| (finalExp.toUInt8 <<< 2) ||| finalMant.toUInt8
    else
      -- Subnormal in e5m2: realExp < -14
      -- Subnormal value = mant * 2^(-16), so we need:
      -- mant = fullMant * 2^(realExp - 23) / 2^(-16) = fullMant >> (7 - realExp)
      let totalShift := (7 - realExp).toNat
      if totalShift >= 25 then
        -- All bits shifted out → zero
        sign8
      else
        let shifted := fullMant >>> totalShift.toUInt32
        let roundBit := if totalShift > 0 then (fullMant >>> (totalShift.toUInt32 - 1)) &&& 1 else 0
        let stickyMask := if totalShift > 1 then (1 <<< (totalShift.toUInt32 - 1)) - 1 else 0
        let stickyBits := fullMant &&& stickyMask
        let rounded := if roundBit == 1 && (stickyBits != 0 || shifted &&& 1 == 1)
          then shifted + 1 else shifted
        -- If rounded up to 4, it becomes the smallest normal (exp=1, mant=0)
        if rounded >= 4 then
          sign8 ||| (1 : UInt8) <<< 2
        else
          sign8 ||| rounded.toUInt8

-- e5m2 encode tests (verified against ml_dtypes)
#guard (Float32.ofBits 0x00000000).toFloat8E5M2Bits == (0 : UInt8)      -- +0
#guard (Float32.ofBits 0x80000000).toFloat8E5M2Bits == (128 : UInt8)    -- -0
#guard (Float32.ofNat 1).toFloat8E5M2Bits == (60 : UInt8)               -- 1.0
#guard (Float32.ofNat 2).toFloat8E5M2Bits == (64 : UInt8)               -- 2.0
#guard (Float32.ofNat 3).toFloat8E5M2Bits == (66 : UInt8)               -- 3.0
#guard (Float32.ofBits 0x47600000).toFloat8E5M2Bits == (123 : UInt8)    -- 57344 (max)
#guard (Float32.ofBits 0x7F800000).toFloat8E5M2Bits == (124 : UInt8)    -- +inf
#guard (Float32.ofBits 0xFF800000).toFloat8E5M2Bits == (252 : UInt8)    -- -inf

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

-- fp16 round-trip. Decode UInt16 as fp16 → Float32 → encode back should give same bits.
-- NaN payloads may not round-trip, so we allow f != f (NaN) as an alternative.
/--
info: Unable to find a counter-example
---
warning: declaration uses 'sorry'
-/
  #guard_msgs in
  example (bits : UInt16) :
    let f := bits.toFloat32FromFloat16
    f.toFloat16Bits == bits ∨ f != f := by plausible

--fp32 0.00005 is subnormal in fp16
-- Numpy encodes it as bits 839 tested using:
-- python3 -c "import numpy as np; x = np.float16(0.00005); print(x, x.view(np.uint16))"
#guard (Float32.ofNat 5 / Float32.ofNat 100000).toFloat16Bits == (839 : UInt16)

-- bf16 subnormal: fp32 1e-40 (0x000116C2) encodes to bf16 bits 1 (verified with ml_dtypes)
#guard (Float32.ofBits 0x000116C2).toBFloat16Bits == (1 : UInt16)
-- bf16 encoding for Nats
#guard (Float32.ofNat 42).toBFloat16Bits == (16936 : UInt16)
-- max safe int for bf16
#guard (Float32.ofNat 256).toBFloat16Bits == (17280 : UInt16)
-- bf16 rounding: 0.1 requires round-to-nearest-even
#guard (Float32.ofNat 1 / Float32.ofNat 10).toBFloat16Bits == (15821 : UInt16)

-- bf16 round-trip: encode then decode should give original value
#guard (16936 : UInt16).toFloat32FromBFloat16 == Float32.ofNat 42

-- bf16 negative value: -42 encodes to 49704, decodes back to -42.0
#guard (49704 : UInt16).toFloat32FromBFloat16 == Float32.ofBits 0xC2280000
-- bf16 subnormal decode: bits 1 pads with zeros to fp32
#guard (1 : UInt16).toFloat32FromBFloat16 == Float32.ofBits 0x00010000
-- bf16 overflow: max fp32 rounds up to infinity (verified with ml_dtypes)
#guard (Float32.ofBits 0x7F7FFFFF).toBFloat16Bits == (32640 : UInt16)


-- Property: bf16 round-trip. Encode UInt16 as bf16 -> decode to Float32 -> encode back.
-- Should give same bits (NaN may not round-trip via ==).
-- Note: NaN patterns pass via f != f escape without testing encode/decode consistency.
-- This is expected because Lean normalizes all NaN to 0x7FC00000 via Float32.toBits, so NaN bits cannot round-trip through
-- fp32 regardless of our encode/decode logic.
/--
info: Unable to find a counter-example
---
warning: declaration uses 'sorry'
-/
#guard_msgs in
  example (bits : UInt16) :
    let f := bits.toFloat32FromBFloat16
    f.toBFloat16Bits == bits ∨ f != f := by plausible



-- fp8e4m3 decode guards verified against ml_dtypes
#guard (0 : UInt8).toFloat32FromFloat8E4M3 == Float32.ofBits 0x00000000       -- +0
#guard (128 : UInt8).toFloat32FromFloat8E4M3 == Float32.ofBits 0x80000000     -- -0
#guard (56 : UInt8).toFloat32FromFloat8E4M3 == 1.0                            -- 1.0
#guard (184 : UInt8).toFloat32FromFloat8E4M3 == -1.0                          -- -1.0
#guard (64 : UInt8).toFloat32FromFloat8E4M3 == 2.0                            -- 2.0
#guard (48 : UInt8).toFloat32FromFloat8E4M3 == 0.5                            -- 0.5
#guard (126 : UInt8).toFloat32FromFloat8E4M3 == Float32.ofBits 0x43E00000     -- max (448.0)
#guard (254 : UInt8).toFloat32FromFloat8E4M3 == Float32.ofBits 0xC3E00000     -- -max (-448.0)
#guard (1 : UInt8).toFloat32FromFloat8E4M3 == Float32.ofBits 0x3B000000       -- smallest subnormal
#guard (8 : UInt8).toFloat32FromFloat8E4M3 == Float32.ofBits 0x3C800000       -- smallest normal
#guard (60 : UInt8).toFloat32FromFloat8E4M3 == 1.5                            -- 1.5
-- NaN: f != f is the standard NaN check
#guard (127 : UInt8).toFloat32FromFloat8E4M3 !=  (127 : UInt8).toFloat32FromFloat8E4M3  -- +NaN
#guard (255 : UInt8).toFloat32FromFloat8E4M3 !=  (255 : UInt8).toFloat32FromFloat8E4M3  -- -NaN

-- fp8e4m3 encode guards
#guard (Float32.ofBits 0x00000000).toFloat8E4M3Bits == (0 : UInt8)      -- +0
#guard (Float32.ofBits 0x80000000).toFloat8E4M3Bits == (128 : UInt8)    -- -0
#guard (Float32.ofNat 1).toFloat8E4M3Bits == (56 : UInt8)               -- 1.0
#guard (Float32.ofNat 2).toFloat8E4M3Bits == (64 : UInt8)               -- 2.0
#guard (Float32.ofNat 42).toFloat8E4M3Bits == (98 : UInt8)              -- 42 -> 40.0
#guard (Float32.ofBits 0x43E00000).toFloat8E4M3Bits == (126 : UInt8)    -- 448.0 (max)
#guard (Float32.ofBits 0x7F800000).toFloat8E4M3Bits == (127 : UInt8)    -- +inf -> NaN
#guard (Float32.ofBits 0xFF800000).toFloat8E4M3Bits == (255 : UInt8)    -- -inf -> -NaN

-- Float32.toInt handles inf/NaN correctly (matches numpy's int64 saturation)
#guard (Float32.ofBits 0xFF800000).toInt == -0x8000000000000000  -- -inf -> INT64_MIN
#guard (Float32.ofBits 0x7F800000).toInt == 0x7FFFFFFFFFFFFFFF   -- +inf -> INT64_MAX
#guard Float32.quietNaN.toInt == 0  -- NaN -> 0


#guard (Float32.ofBits 0x7F800000).toNat == 0xFFFFFFFFFFFFFFFF  -- +inf -> UINT64_MAX
#guard (Float32.ofBits 0xFF800000).toNat == 0  -- -inf -> 0
#guard Float32.quietNaN.toNat == 0  -- NaN -> 0

#guard (Float.ofBits 0x7FF0000000000000).toNat == 0xFFFFFFFFFFFFFFFF  -- +inf -> UINT64_MAX
#guard (Float.ofBits 0xFFF0000000000000).toNat == 0                    -- -inf -> 0
#guard Float.quietNaN.toNat == 0                                       -- NaN -> 0
#guard (Float.ofBits 0xFFF0000000000000).toInt == -0x8000000000000000  -- -inf -> INT64_MIN
#guard (Float.ofBits 0x7FF0000000000000).toInt == 0x7FFFFFFFFFFFFFFF   -- +inf -> INT64_MAX
#guard Float.quietNaN.toInt == 0                                       -- NaN -> 0


-- e4m3 round-trip: Encode as e4m3 -> decode to fp32 -> encode back
-- Should give same bits (NaN may not round-trip via == so we use f != f escape)
/--
info: Unable to find a counter-example
---
warning: declaration uses 'sorry'
-/
#guard_msgs in
  example (bits : UInt8) :
    let f := bits.toFloat32FromFloat8E4M3
    f.toFloat8E4M3Bits == bits ∨ f != f := by plausible


-- Property: e5m2 round-trip (except NaN)
-- Decode -> encode should give same bit
/--
info: Unable to find a counter-example
---
warning: declaration uses 'sorry'
-/
#guard_msgs in
  example (bits : UInt8) :
    let f := bits.toFloat32FromFloat8E5M2
    f.toFloat8E5M2Bits == bits ∨ f != f := by plausible

end Test

end TensorLib
