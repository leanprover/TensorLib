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

def sizedStrides (dtype : Dtype) (s : Shape) : Strides := List.map (fun x => x * dtype.itemsize) s.cUnitStrides

end Dtype
end TensorLib
