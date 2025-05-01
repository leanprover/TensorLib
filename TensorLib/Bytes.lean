/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/

import TensorLib.Common

namespace TensorLib

-- We cast between UIntX and ByteArray without going through BitVec, which is represented
-- as a Nat at runtime. We'll have tests for these in our ByteArray extension module where we
-- have conversions back and forth between LE ByteArrays and UIntX.
def _root_.UInt8.toLEByteArray (n : UInt8) : ByteArray := ByteArray.mk #[n]

def _root_.UInt16.toLEByteArray (n : UInt16) : ByteArray :=
  let b0 := (n      ).toUInt8
  let b1 := (n >>> 8).toUInt8
  ByteArray.mk #[b0, b1]

def _root_.UInt16.toLEByteArray' (n : UInt16) : ByteArray :=
  let b0 := (n      ).toUInt8
  let b1 := (n >>> 8).toUInt8
  ByteArray.mk #[b0, b1]

def _root_.UInt32.toLEByteArray (n : UInt32) : ByteArray :=
  let b0 := (n       ).toUInt8
  let b1 := (n >>>  8).toUInt8
  let b2 := (n >>> 16).toUInt8
  let b3 := (n >>> 24).toUInt8
  ByteArray.mk #[b0, b1, b2, b3]

def _root_.UInt64.toLEByteArray (n : UInt64) : ByteArray :=
  let b0 := (n       ).toUInt8
  let b1 := (n >>>  8).toUInt8
  let b2 := (n >>> 16).toUInt8
  let b3 := (n >>> 24).toUInt8
  let b4 := (n >>> 32).toUInt8
  let b5 := (n >>> 40).toUInt8
  let b6 := (n >>> 48).toUInt8
  let b7 := (n >>> 56).toUInt8
  ByteArray.mk #[b0, b1, b2, b3, b4, b5, b6, b7]

def _root_.Int8.toLEByteArray (n : Int8) : ByteArray := n.toUInt8.toLEByteArray
def _root_.Int16.toLEByteArray (n : Int16) : ByteArray := n.toUInt16.toLEByteArray
def _root_.Int32.toLEByteArray (n : Int32) : ByteArray := n.toUInt32.toLEByteArray
def _root_.Int64.toLEByteArray (n : Int64) : ByteArray := n.toUInt64.toLEByteArray

end TensorLib
