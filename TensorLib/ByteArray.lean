/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/

import Plausible
import TensorLib.Bytes
import TensorLib.Common

namespace TensorLib

/-- Interpret a `ByteArray` of size 4 as a little-endian `UInt32`. Missing from Lean stdlib. -/
def _root_.ByteArray.toUInt32LE (bs : ByteArray) : Err UInt32 :=
  if bs.size != 4 then throw "Expected size 4 byte array" else
  return (bs.get! 0).toUInt32          |||
         (bs.get! 1).toUInt32 <<< 0x8  |||
         (bs.get! 2).toUInt32 <<< 0x10 |||
         (bs.get! 3).toUInt32 <<< 0x18

def _root_.ByteArray.toUInt32LE! (bs : ByteArray) : UInt32 := get! bs.toUInt32LE

section Test

private def roundTripUInt32LE (x : UInt32) : Bool :=
  let c := bitVecToLEByteArray 32 x.toBitVec
  let x' := c.toUInt32LE!
  x == x'

#guard roundTripUInt32LE 0xFFFF
#guard roundTripUInt32LE 0x1010

/--
info: Unable to find a counter-example
---
warning: declaration uses 'sorry'
-/
#guard_msgs in
example (x : UInt32) : roundTripUInt32LE x := by plausible

end Test

end TensorLib
