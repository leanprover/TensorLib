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
  toString x := (repr x).pretty

def isMultiByte (x : Name) : Bool := match x with
| bool | int8 | uint8 => false
| _ => true

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

def byteOrderOk (x : Dtype) : Prop := !x.name.isMultiByte || (x.name.isMultiByte && x.order.isMultiByte)

def itemsize (t : Dtype) := t.name.itemsize

def sizedStrides (dtype : Dtype) (s : Shape) : Strides := List.map (fun x => x * dtype.itemsize) s.unitStrides

def isInt (dtype : Dtype) : Bool := dtype.name.isInt
def isUint (dtype : Dtype) : Bool := dtype.name.isUint
def isIntLike (dtype : Dtype) : Bool := dtype.isInt || dtype.isUint

def int8 : Dtype := Dtype.mk Dtype.Name.int8 ByteOrder.littleEndian
def uint8 : Dtype := Dtype.mk Dtype.Name.uint8 ByteOrder.littleEndian
def int64 : Dtype := Dtype.mk Dtype.Name.int64 ByteOrder.littleEndian
def uint64 : Dtype := Dtype.mk Dtype.Name.uint64 ByteOrder.littleEndian
def float64 : Dtype := Dtype.mk Dtype.Name.float64 ByteOrder.littleEndian

end Dtype
end TensorLib
