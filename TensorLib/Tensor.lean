import TensorLib.Common
import TensorLib.Dtype
import TensorLib.Npy

namespace TensorLib

local instance ByteArrayRepr : Repr ByteArray where
  reprPrec x _ :=
    let s := toString x.size
    s!"ByteArray of size {s}"

/-!
A Tensor is the data bytes, along with all metadata required to do
efficient computation. The `startIndex`, `unitstrides` are inferred
from parsing or computed during a view creation.

Note that this representation is slightly different from the C version in NumPy source.

1. One difference is that we maintain "unit"-strides rather than strides. A unit stride
is just the stride divided by the datatype size. This makes indexing and iterating more
straightforward in my opinion. When you need to jump, you just need to remember to multiply
the number of slots by the datatype size.

2. Another is that we maintain a starting index into the array. Thus, if
we reverse a 1-D array, we keep the same ByteArray and update the start index. In C,
the `data` field is a pointer to a char array, and thus that can serve as the starting
point directly. This took me a while to figure out, so let me document an example

# x = np.arange(6, dtype='uint8')
# y = x[::-1]
# np.array_equal(y.base, x)
True

# x.ctypes.data
105553176936576

# y.ctypes.data
105553176936581

# y.ctypes.data - x.ctypes.data
5

Note that `x.data` and `y.data` exist, but are abstract types that, while there are addresses printed
with them, don't have this obvious behavior.

# y.base.data
<memory at 0x111694f40>

# x.data
<memory at 0x111694dc0>

y.base and x are the same so I don't know what the non-ctypes `data` field actually represents.
They certainly don't have the offset like the ctypes version.

The data order is used in Numpy for deciding if/when to make copies. There's a nice
comment about how to decide,

https://github.com/numpy/numpy/blob/main/numpy/_core/src/multiarray/flagsobject.c#L92-L114

   The traditional rule is that for an array to be flagged as C contiguous,
   the following must hold:

   strides[-1] == itemsize
   strides[i] == shape[i+1] * strides[i + 1]

   And for an array to be flagged as F contiguous, the obvious reversal:

   strides[0] == itemsize
   strides[i] == shape[i - 1] * strides[i - 1]

   According to these rules, a 0- or 1-dimensional array is either both
   C- and F-contiguous, or neither; and an array with 2+ dimensions
   can be C- or F- contiguous, or neither, but not both (unless it has only
   a single element).
   We correct this, however.  When a dimension has length 1, its stride is
   never used and thus has no effect on the  memory layout.
   The above rules thus only apply when ignoring all size 1 dimensions.

If we decide to do reference counting and copying as in NumPy, we will
need this info, but for now we will copy whenever we update the array.

-/
-- TODO: Add a `base` field to track aliasing? NumPy does this and it may make sense for us.
-- TODO: Do we want this to be inductive to handle array scalars? https://numpy.org/doc/stable/reference/arrays.scalars.html#arrays-scalars
--       Will force those into this type for now, but it seems wasteful.
--       NumPy has a bunch of special handling for array scalars.
structure Tensor where
  dtype : Dtype
  dataOrder : DataOrder := DataOrder.C
  shape : Shape
  data : ByteArray
  startIndex : Nat := 0 -- Pointer to the first byte of ndarray data. This is implicit in the `data` pointer in numpy.
  unitStrides : Strides := shape.unitStrides dataOrder
  deriving Repr, Inhabited

namespace Tensor

private def dtypeOfNpy (dtype : Npy.Dtype) : Err Dtype := do
  let order <- match dtype.order with
  | .bigEndian => .ok .bigEndian
  | .littleEndian => .ok .littleEndian
  | .notApplicable => .ok .oneByte
  | .native => .error "native byte order not supported"
  .ok $ Dtype.mk dtype.name order

private def dataOfNpy (arr : Npy.Ndarray) : ByteArray :=
  let dst := ByteArray.mkEmpty arr.nbytes
  arr.data.copySlice arr.startIndex dst 0 arr.nbytes

/-
Makes a copy of the data, dropping the header and padding.
Probably not a great choice, but sticking with it for now.
I want to avoid writing .npy files with wrong header data.
-/
def ofNpy (arr : Npy.Ndarray) : Err Tensor := do
  let dtype <- dtypeOfNpy arr.header.descr
  let dataOrder := arr.header.dataOrder
  let shape := arr.header.shape
  let data := dataOfNpy arr
  let startIndex := 0
  return { dtype, dataOrder, shape, data, startIndex }

private def dtypeToNpy (dtype : Dtype) : Npy.Dtype :=
  let order := match dtype.order with
  | .bigEndian => .bigEndian
  | .littleEndian => .littleEndian
  | .oneByte => .notApplicable
  Npy.Dtype.mk dtype.name order

def toNpy (arr : Tensor) : Npy.Ndarray :=
  let descr := dtypeToNpy arr.dtype
  let dataOrder := arr.dataOrder
  let shape := arr.shape
  let header : Npy.Header := { descr, dataOrder, shape }
  let data := arr.data
  let startIndex := 0
  { header, data, startIndex }

def empty (dtype : Dtype) (shape : Shape) : Tensor :=
  let data := ByteArray.mkEmpty (dtype.itemsize * shape.count)
  { dtype := dtype, shape := shape, data := data }

--! number of dimensions
def ndim (x : Tensor) : ℕ := x.shape.length

--! number of elements
def size (x : Tensor) : ℕ := x.shape.count

--! number of bytes representing each element
def itemsize (x : Tensor) : ℕ := x.dtype.itemsize

--! byte-strides
def strides (x : Tensor) : Strides := x.unitStrides.map (fun v => x.itemsize * v)

--! number of bytes representing the entire tensor
def nbytes (x : Tensor) : ℕ := x.itemsize * x.size


class Element (a : Type) where
  dtype : Dtype
  itemsize : Nat
  ofNat : Nat -> a
  toByteArray (x : a) : ByteArray
  fromByteArray (arr : ByteArray) (startIndex : Nat) : Err a

namespace Element

-- An array-scalar is a box around a scalar with nil shape that can be used for array operations like broadcasting
def arrayScalar [w : Element a] (x : a) : Tensor :=
  { dtype := w.dtype, shape := [], data := w.toByteArray x}

--! An array of the numbers from 0 to n-1
--! https://numpy.org/doc/2.1/reference/generated/numpy.arange.html
def arange (a : Type) [w : Element a] (n : Nat) : Tensor :=
  let data := ByteArray.mkEmpty (n * w.itemsize)
  let foldFn i data :=
    let bytes := w.toByteArray (w.ofNat i)
    ByteArray.copySlice bytes 0 data (i * w.itemsize) w.itemsize
  let data := Nat.fold foldFn n data
  { dtype := w.dtype, shape := [n], data }

instance BV8Native : Element BV8 where
  dtype := Dtype.mk .uint8 .oneByte
  itemsize := 1
  ofNat n := n
  toByteArray (x : BV8) : ByteArray := x.toByteArray
  fromByteArray arr startIndex := ByteArray.toBV8 arr startIndex

instance BV16Little : Element BV16 where
  dtype := Dtype.mk .uint16 .littleEndian
  itemsize := 2
  ofNat n := n
  toByteArray (x : BV16) : ByteArray := x.toByteArray .littleEndian
  fromByteArray arr startIndex := ByteArray.toBV16 arr startIndex .littleEndian

instance BV32Little : Element BV32 where
  dtype := Dtype.mk .uint32 .littleEndian
  itemsize := 4
  ofNat n := n
  toByteArray (x : BV32) : ByteArray := x.toByteArray .littleEndian
  fromByteArray arr startIndex := ByteArray.toBV32 arr startIndex .littleEndian

instance BV64Little : Element BV64 where
  dtype := Dtype.mk .uint64 .littleEndian
  itemsize := 8
  ofNat n := n
  toByteArray (x : BV64) : ByteArray := x.toByteArray .littleEndian
  fromByteArray arr startIndex := ByteArray.toBV64 arr startIndex .littleEndian

#guard (arange BV16 10).size == 10

end Element
end Tensor
end TensorLib
