/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/

import Plausible
import TensorLib.Bytes
import TensorLib.Common

namespace TensorLib

instance : BEq ByteArray where
  beq x y := x.data == y.data

-- We generally have large tensors, so don't show them by default
instance : Repr ByteArray where
  reprPrec x _ :=
    if x.size < 100 then x.toList.repr 100 else
    s!"ByteArray of size {x.size}"

-- How I usually want to use ByteArray.extract
def _root_.ByteArray.sub (a : ByteArray) (start size : Nat) : ByteArray :=
  a.extract start (start + size)

/- The library function has no safe version -/
def _root_.ByteArray.toUInt64LE (bs : ByteArray) : Err UInt64 :=
  if bs.size != 8 then throw "Expected size 8 byte array" else return bs.toUInt64LE!

/- The library function has no safe version -/
def _root_.ByteArray.toUInt64BE (bs : ByteArray) : Err UInt64 :=
  if bs.size != 8 then throw "Expected size 8 byte array" else return bs.toUInt64BE!

/-- Interpret a `ByteArray` of size 4 as a little-endian `UInt32`. Missing from Lean stdlib. -/
def _root_.ByteArray.toUInt32LE (bs : ByteArray) : Err UInt32 :=
  if bs.size != 4 then throw "Expected size 4 byte array" else
  return (bs.get! 0).toUInt32          |||
         (bs.get! 1).toUInt32 <<< 0x8  |||
         (bs.get! 2).toUInt32 <<< 0x10 |||
         (bs.get! 3).toUInt32 <<< 0x18

/-- Interpret a `ByteArray` of size 4 as a little-endian `UInt32`. Missing from Lean stdlib. -/
def _root_.ByteArray.toUInt32BE (bs : ByteArray) : Err UInt32 :=
  if bs.size != 4 then throw "Expected size 4 byte array" else
  return (bs.get! 3).toUInt32          |||
         (bs.get! 2).toUInt32 <<< 0x8  |||
         (bs.get! 1).toUInt32 <<< 0x10 |||
         (bs.get! 0).toUInt32 <<< 0x18

def _root_.ByteArray.toUInt32LE! (bs : ByteArray) : UInt32 := get! bs.toUInt32LE
def _root_.ByteArray.toUInt32BE! (bs : ByteArray) : UInt32 := get! bs.toUInt32BE

/-- Interpret a `ByteArray` of size 2 as a little-endian `UInt16`. Missing from Lean stdlib. -/
def _root_.ByteArray.toUInt16LE (bs : ByteArray) : Err UInt16 :=
  if bs.size != 2 then throw "Expected size 2 byte array" else
  return (bs.get! 0).toUInt16          |||
         (bs.get! 1).toUInt16 <<< 0x8

/-- Interpret a `ByteArray` of size 4 as a little-endian `UInt16`. Missing from Lean stdlib. -/
def _root_.ByteArray.toUInt16BE (bs : ByteArray) : Err UInt16 :=
  if bs.size != 2 then throw "Expected size 2 byte array" else
  return (bs.get! 1).toUInt16          |||
         (bs.get! 0).toUInt16 <<< 0x8

def _root_.ByteArray.toUInt16LE! (bs : ByteArray) : UInt16 := get! bs.toUInt16LE
def _root_.ByteArray.toUInt16BE! (bs : ByteArray) : UInt16 := get! bs.toUInt16BE

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

def _root_.ByteArray.toBool (arr : ByteArray) : Bool := arr.data.any fun v => v != 0

def _root_.ByteArray.take (arr : ByteArray) (n : Nat) : ByteArray :=
  ByteArray.mk (arr.data.take n)

def _root_.ByteArray.drop (arr : ByteArray) (n : Nat) : ByteArray :=
  ByteArray.mk (arr.data.drop n)

def _root_.ByteArray.readUInt32 (arr : ByteArray) (offset : Nat) : UInt32 :=
  if arr.size < offset + 4 then 0 else
  (arr.sub offset 4).toUInt32LE!

def _root_.ByteArray.readUInt64 (arr : ByteArray) (offset : Nat) : UInt64 :=
  if arr.size < offset + 8 then 0 else
  (arr.sub offset 8).toUInt64LE!

section Test

private def roundTripUInt32LE (x : UInt32) : Bool := (toLEByteArray x).toUInt32LE! == x
private def roundTripUInt32BE (x : UInt32) : Bool := (toBEByteArray x).toUInt32BE! == x

#guard roundTripUInt32LE 0xFFFF
#guard roundTripUInt32LE 0x1010
#guard roundTripUInt32BE 0xFFFF
#guard roundTripUInt32BE 0x1010

/--
info: Unable to find a counter-example
---
warning: declaration uses 'sorry'
-/
#guard_msgs in
example (x : UInt32) : roundTripUInt32LE x && roundTripUInt32BE x := by plausible

end Test

instance : ToLEByteArray (BitVec n) where
  toLEByteArray bv := Id.run do
    let value := bv.toNat
    let numBytes := (n + 7) / 8
    let mut ba : ByteArray := ByteArray.emptyWithCapacity numBytes
    let mut remaining := value
    for _ in [:numBytes] do
      let byte : UInt8 := (remaining % 256).toUInt8
      remaining := remaining / 256
      ba := ba.push byte
    ba

instance : ToBEByteArray (BitVec n) where
  toBEByteArray bv := (toLEByteArray bv).reverse

#guard toBEByteArray (BitVec.ofNat 8 0xba) == ByteArray.mk #[0xba]
#guard toBEByteArray (BitVec.ofNat 24 0xfedcba) == ByteArray.mk #[0xfe, 0xdc, 0xba]
#guard toBEByteArray (BitVec.ofNat 24 0xabcdef) == ByteArray.mk #[0xab, 0xcd, 0xef]

def _root_.ByteArray.toBitVecBE (arr : ByteArray) (n : Nat) : BitVec n := Id.run do
  let bytesNeeded : Nat := (n + 7) / 8
  let mut result : Nat := 0
  let bytesToProcess := min bytesNeeded arr.size
  for i in [:bytesToProcess] do
    let byte : Nat := arr[i]!.toNat
    result := (result <<< 8) ||| byte
  let mask : Nat := if n == 0 then 0 else (1 <<< n) - 1
  let finalValue := result &&& mask
  BitVec.ofNat n finalValue

def _root_.ByteArray.toBitVecLE (arr : ByteArray) (n : Nat) : BitVec n := arr.reverse.toBitVecBE n

section Test
open Plausible

-- TODO: The BitVec/ByteArray round trips only work right now when the sizes line up. Relax this.

#guard
  let arr := ByteArray.mk #[0xff, 0xee, 0xdd]
  toBEByteArray (arr.toBitVecBE 24) == arr

#guard
  let arr := ByteArray.mk #[0xff, 0xee, 0xdd]
  toLEByteArray (arr.toBitVecLE 24) == arr

private local instance : Shrinkable ByteArray where

private local instance : SampleableExt ByteArray :=
  SampleableExt.mkSelfContained do
    let data ← SampleableExt.interpSample (Array UInt8)
    return ByteArray.mk data

/--
info: Unable to find a counter-example
---
warning: declaration uses 'sorry'
-/
#guard_msgs in
example (arr : ByteArray) :
  let bv := arr.toBitVecBE (8 * arr.size)
  let v := toBEByteArray bv
  arr == v := by plausible

/--
info: Unable to find a counter-example
---
warning: declaration uses 'sorry'
-/
#guard_msgs in
example (arr : ByteArray) :
  let bv := arr.toBitVecLE (8 * arr.size)
  let v := toLEByteArray bv
  arr == v := by plausible

end Test

end TensorLib
