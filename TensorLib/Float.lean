/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
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
-- Add 1 to the mantissa length because of the implicit leading 1
def maxSafeNatForFloat32 : Nat := Nat.pow 2 (float32MantissaBits + 1)
def maxSafeNatForFloat64 : Nat := Nat.pow 2 (float64MantissaBits + 1)

def _root_.Int.toFloat32 (n : Int) : Float32 := Float32.ofInt n
def _root_.Int.toFloat64 (n : Int) : Float := Float.ofInt n

def _root_.Float.toLEByteArray (f : Float) : ByteArray := f.toBits.toLEByteArray
def _root_.Float32.toLEByteArray (f : Float32) : ByteArray := f.toBits.toLEByteArray

def Float32.ofLEByteArray (arr : ByteArray) : Err Float32 := do
  return Float32.ofBits (<- arr.toUInt32LE)

def Float32.ofLEByteArray! (arr : ByteArray) : Float32 := get! $ Float32.ofLEByteArray arr

def Float.ofLEByteArray (arr : ByteArray) : Err Float := do
  return Float.ofBits (<- arr.toUInt64LE)

def Float.ofLEByteArray! (arr : ByteArray) : Float := get! $ Float.ofLEByteArray arr

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

#guard Float.quietNaN.toInt == 0

section Test

#guard (
  let n : UInt64 := 0x3FFAB851EB851EB8
  do
    let arr := n.toLEByteArray
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
