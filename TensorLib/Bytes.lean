/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/

import TensorLib.Common
namespace TensorLib

class ToLEByteArray (a : Type) where
  toLEByteArray : a -> ByteArray

export ToLEByteArray(toLEByteArray)

class ToBEByteArray (a : Type) where
  toBEByteArray : a -> ByteArray

export ToBEByteArray(toBEByteArray)

instance : ToLEByteArray ByteArray where
  toLEByteArray arr := arr

instance : ToBEByteArray ByteArray where
  toBEByteArray arr := arr

-- We cast between UIntX and ByteArray without going through BitVec, which is represented
-- as a Nat at runtime. We'll have tests for these in our ByteArray extension module where we
-- have conversions back and forth between LE ByteArrays and UIntX.
instance : ToLEByteArray UInt8 where
  toLEByteArray n := ByteArray.mk #[n]

instance : ToBEByteArray UInt8 where
  toBEByteArray n := ByteArray.mk #[n]

instance : ToLEByteArray Int8 where
  toLEByteArray n := toLEByteArray n.toUInt8

instance : ToBEByteArray Int8 where
  toBEByteArray n := toBEByteArray n.toUInt8

instance : ToLEByteArray UInt16 where
  toLEByteArray n :=
    let b0 := (n      ).toUInt8
    let b1 := (n >>> 8).toUInt8
    ByteArray.mk #[b0, b1]

instance : ToBEByteArray UInt16 where
  toBEByteArray n :=
    let b0 := (n      ).toUInt8
    let b1 := (n >>> 8).toUInt8
    ByteArray.mk #[b1, b0]

instance : ToLEByteArray Int16 where
  toLEByteArray n := toLEByteArray n.toUInt16

instance : ToBEByteArray Int16 where
  toBEByteArray n := toBEByteArray n.toUInt16

instance : ToLEByteArray UInt32 where
  toLEByteArray n :=
  let b0 := (n       ).toUInt8
  let b1 := (n >>>  8).toUInt8
  let b2 := (n >>> 16).toUInt8
  let b3 := (n >>> 24).toUInt8
  ByteArray.mk #[b0, b1, b2, b3]

instance : ToBEByteArray UInt32 where
  toBEByteArray n :=
    let b0 := (n       ).toUInt8
    let b1 := (n >>>  8).toUInt8
    let b2 := (n >>> 16).toUInt8
    let b3 := (n >>> 24).toUInt8
    ByteArray.mk #[b3, b2, b1, b0]

instance : ToLEByteArray Int32 where
  toLEByteArray n := toLEByteArray n.toUInt32

instance : ToBEByteArray Int32 where
  toBEByteArray n := toBEByteArray n.toUInt32

instance : ToLEByteArray UInt64 where
  toLEByteArray n :=
    let b0 := (n       ).toUInt8
    let b1 := (n >>>  8).toUInt8
    let b2 := (n >>> 16).toUInt8
    let b3 := (n >>> 24).toUInt8
    let b4 := (n >>> 32).toUInt8
    let b5 := (n >>> 40).toUInt8
    let b6 := (n >>> 48).toUInt8
    let b7 := (n >>> 56).toUInt8
    ByteArray.mk #[b0, b1, b2, b3, b4, b5, b6, b7]

instance : ToBEByteArray UInt64 where
  toBEByteArray n :=
    let b0 := (n       ).toUInt8
    let b1 := (n >>>  8).toUInt8
    let b2 := (n >>> 16).toUInt8
    let b3 := (n >>> 24).toUInt8
    let b4 := (n >>> 32).toUInt8
    let b5 := (n >>> 40).toUInt8
    let b6 := (n >>> 48).toUInt8
    let b7 := (n >>> 56).toUInt8
    ByteArray.mk #[b7, b6, b5, b4, b3, b2, b1, b0]

instance : ToLEByteArray Int64 where
  toLEByteArray n := toLEByteArray n.toUInt64

instance : ToBEByteArray Int64 where
  toBEByteArray n := toBEByteArray n.toUInt64

end TensorLib
