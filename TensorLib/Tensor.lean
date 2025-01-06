/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/

import Batteries.Data.List
import TensorLib.Common
import TensorLib.Dtype
import TensorLib.Npy

namespace TensorLib


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

If we decide to do reference counting and copying as in NumPy, we will
need this info, but for now we will copy whenever we update the array.

-/
-- TODO: Add a `base` field to track aliasing? NumPy does this and it may make sense for us.
-- TODO: Do we want this to be inductive to handle array scalars? https://numpy.org/doc/stable/reference/arrays.scalars.html#arrays-scalars
--       Will force those into this type for now, but it seems wasteful.
--       NumPy has a bunch of special handling for array scalars.
-- TODO: I'm really not sure what to do with the data order here. Logically, the strides are
--       enough to navigate the tensor. E.g. x.T just reverses the strides, and everything works fine.
--       NumPy though, swaps the data order flag during a transpose as well. It is used for optimizations,
--       especially avoiding copies. We should get a good handle on this logic and either properly use
--       the data order field or remove it entirely.

structure Tensor where
  dtype : Dtype
  shape : Shape
  data : ByteArray
  startIndex : Nat := 0 -- Pointer to the first byte of ndarray data. This is implicit in the `data` pointer in numpy.
  unitStrides : Strides := shape.unitStrides
  deriving Repr, Inhabited
namespace Tensor

/-!
An tensor is trivially reshapable if it is contiguous with non-negative strides.
-/
def isTriviallyReshapable (arr : Tensor) : Bool :=
  arr.startIndex == 0
  && arr.unitStrides == arr.shape.unitStrides

def empty (dtype : Dtype) (shape : Shape) : Tensor :=
  let data := ByteArray.mkEmpty (dtype.itemsize * shape.count)
  { dtype := dtype, shape := shape, data := data }

def zeros (dtype : Dtype) (shape : Shape) : Tensor := Id.run do
  let size := dtype.itemsize * shape.count
  let mut data := ByteArray.mkEmpty size
  for _ in [0:size] do
    data := data.push 0
  { dtype := dtype, shape := shape, data := data }

def ones (dtype : Dtype) (shape : Shape) : Tensor := Id.run do
  let size := dtype.itemsize * shape.count
  let itemsize := dtype.itemsize
  let mut data := ByteArray.mkEmpty size
  for i in [0:size] do
    let byte := match dtype.order with
    | .oneByte => 1
    | .littleEndian => if i.mod itemsize == 0 then 1 else 0
    | .bigEndian => if i.mod itemsize == itemsize - 1 then 1 else 0
    data := data.push byte
  { dtype := dtype, shape := shape, data := data }

def byteOrder (arr : Tensor) : ByteOrder := arr.dtype.order

--! number of dimensions
def ndim (x : Tensor) : Nat := x.shape.ndim

--! number of elements
def size (x : Tensor) : Nat := x.shape.count

--! number of bytes representing each element
def itemsize (x : Tensor) : Nat := x.dtype.itemsize

--! byte-strides
def strides (x : Tensor) : Strides := x.unitStrides.map (fun v => x.itemsize * v)

--! Get the offset corresponding to a DimIndex
def dimIndexToOffset (x : Tensor) (i : DimIndex) : Int :=
  Shape.dimIndexToOffset x.strides i

--! Get the starting byte corresponding to a DimIndex
def dimIndexToPosition (x : Tensor) (i : DimIndex) : Nat :=
  (x.startIndex + (x.dimIndexToOffset i)).toNat

--! number of bytes representing the entire tensor
def nbytes (x : Tensor) : Nat := x.itemsize * x.size

def isIntLike (x : Tensor) : Bool := x.dtype.isIntLike

def dimIndexInRange (arr : Tensor) (dimIndex : DimIndex) : Bool := arr.shape.dimIndexInRange dimIndex

def byteArrayAtDimIndex (arr : Tensor) (dimIndex : DimIndex) : Err ByteArray := do
  if !arr.dimIndexInRange dimIndex then .error "index is incompatible with tensor shape" else
  let posn := arr.dimIndexToPosition dimIndex
  .ok $ arr.data.extract posn (posn + arr.itemsize)

def setByteArrayAtDimIndex (arr : Tensor) (dimIndex : DimIndex) (bytes : ByteArray) : Err Tensor := do
  if !arr.dimIndexInRange dimIndex then .error "index is incompatible with tensor shape" else
  if arr.itemsize != bytes.size then .error "byte size mismatch" else
  let posn := arr.dimIndexToPosition dimIndex
  .ok $ { arr with data := bytes.copySlice 0 arr.data posn bytes.size }

/-!
Return the integer at the dimIndex. This is useful, for example, in advanced indexing
where we must have an int/uint Tensor as an argument.
-/
def intAtDimIndex (arr : Tensor) (dimIndex : DimIndex) : Err Int := do
  if !arr.isIntLike then .error "natAt expects an int tensor" else
  let bytes <- byteArrayAtDimIndex arr dimIndex
  .ok $ arr.byteOrder.bytesToInt bytes

/-!
Copy a Tensor's data to new, contiguous storage.

Note that this doesn't just do a `ByteArray` copy. It walks over `arr` according
to its strides and selects the elements, so it works on views of other tensors with
non-contiguous data.
-/
def copy (arr : Tensor) : Tensor := Id.run do
  let itemsize := arr.dtype.itemsize
  let mut data := ByteArray.mkEmpty arr.nbytes
  let iter := DimsIter.make arr.shape
  for dimIndex in iter do
    let posn := arr.dimIndexToPosition dimIndex
    for j in [0:itemsize] do
      let b := arr.data.get! (posn + j)
      -- Leaving this here for posterity. Random access `set!` doesn't work on
      -- arrays created using `ByteArray.empty`, even if the initial size is greater
      -- than the argument to `set!`. It just quietly does nothing.
      -- data := data.set! i b
      data := data.push b
  let arr := {
     dtype := arr.dtype,
     shape := arr.shape,
     data
  }
  return arr

/-
Reshaping is surprisingly tricky. These are the main methods in NumPy.

  https://github.com/numpy/numpy/blob/main/numpy/_core/src/multiarray/shape.c#L187-L189
  https://github.com/numpy/numpy/blob/main/numpy/_core/src/multiarray/shape.c#L195-L197
  https://github.com/numpy/numpy/blob/main/numpy/_core/src/multiarray/shape.c#L347
    let olds := x.shape

It's a couple hundred lines of C code with many corner cases. For example, it gets
hard when the array is already a view, especially one that is already non-contiguous.
If the array is contiguous, and strides are positive, things are much easier;
we can just write down the new shape and compute the new strides. Things that complicate
the picture, and require all the logic in the code above:

1. Axes of size 1 have a non-0 stride, but it shouldn't be used in new stride calculations.
2. If the array is not contiguous, e.g. created with a non-unit `step` in a slice, we
   may or may not be able to represent it without copying.
3. Even if the array is contiguous, with negative strides, e.g. created with a negative `step` in a slice, we
   may or may not be able to represent it without copying. For example, if some strides are positive and
   some are negative, it can require a copy to reshape.
4. Data ordering (C vs Fortran, row major vs column major) goes into the calculation of whether an array
   is contiguous or not. The same strides on the same data can be contiguous or not depending on the data ordering.

Examples:

Reshapes can require a copy. For example, when we get the data out of order via
reverses and reshapes, flattening it again will require a copy.

# x = np.arange(6)

# x.reshape(3, 2).base is x
True

# x.reshape(3, 2)[::-1].base is x
True

# x.reshape(3, 2)[::-1].reshape(6)
array([4, 5, 2, 3, 0, 1])

# x.reshape(3, 2)[::-1].reshape(6).base is x
False

Clearly a simple list of numbers for strides can't jump around the original list
to capture that pattern. My guess is that there is we can figure out if we need
a copy by looking at startPosition, shape, and strides.
-/
private def copyAndReshape (arr : Tensor) (shape : Shape) : Err Tensor :=
  if arr.shape.count != shape.count then .error "Incompatible shapes" else
  let arr := arr.copy
  .ok { arr with shape, unitStrides := shape.unitStrides }

def copyAndReshape! (arr : Tensor) (shape : Shape) : Tensor :=
  get! (copyAndReshape arr shape)

def reshape (arr : Tensor) (shape : Shape) : Err Tensor :=
  if arr.shape.count != shape.count then .error "Incompatible shapes" else
  if arr.isTriviallyReshapable
  then .ok { arr with shape, unitStrides := shape.unitStrides }
  else copyAndReshape arr shape

def reshape! (arr : Tensor) (shape : Shape) : Tensor := get! $ reshape arr shape

/-
NumPy allows you to transpose flexibly on the axes, e.g. allowing arbitrary
reorders. This is just the default, which reverses the dimensions.
It is also extremely conservative when it comes to needing copies.
-/
def transpose (arr : Tensor) : Tensor :=
  let shape : Shape := arr.shape.map List.reverse
  if arr.isTriviallyReshapable
  then { arr with shape, unitStrides := shape.unitStrides }
  else
    let arr := arr.copy
    { arr with shape, unitStrides := shape.unitStrides }

class Element (a : Type) where
  dtype : Dtype
  itemsize : Nat
  ofNat : Nat -> a
  toByteArray (x : a) : ByteArray
  fromByteArray (arr : ByteArray) (startIndex : Nat) : Err a

namespace Element

-- An array-scalar is a box around a scalar with nil shape that can be used for array operations like broadcasting
def arrayScalar [w : Element a] (x : a) : Tensor :=
  { dtype := w.dtype, shape := Shape.empty, data := w.toByteArray x}

--! An array of the numbers from 0 to n-1
--! https://numpy.org/doc/2.1/reference/generated/numpy.arange.html
def arange (a : Type) [w : Element a] (n : Nat) : Tensor := Id.run do
  let mut data := ByteArray.mkEmpty (n * w.itemsize)
  for i in [0:n] do
    let bytes := w.toByteArray (w.ofNat i)
    data := ByteArray.copySlice bytes 0 data (i * w.itemsize) w.itemsize
  { dtype := w.dtype, shape := Shape.mk [n], data }

-- This is a blind index into the array, disregarding the shape.
def getPosition [typ : Element a] (x : Tensor) (position : Nat) : Err a :=
  if typ.itemsize != x.itemsize then .error "byte size mismatch" else -- TODO: Lift this check out so we only do it once
  typ.fromByteArray x.data (position * typ.itemsize)

def setPosition [typ : Element a] (x : Tensor) (n : Nat) (v : a): Err Tensor :=
  let itemsize := typ.itemsize
  if itemsize != x.itemsize then .error "byte size mismatch" else -- TODO: Lift this check out so we only do it once
  let bytes := typ.toByteArray v
  let posn := n * itemsize
  .ok { x with data := bytes.copySlice 0 x.data posn itemsize true }

def ofList (typ : Element a) (xs : List a) : Tensor := Id.run do
  let arr := Tensor.zeros typ.dtype (Shape.mk [xs.length])
  let mut data := arr.data
  let mut posn := 0
  for x in xs do
    let v := typ.toByteArray x
    data := v.copySlice 0 data posn typ.itemsize
    posn := posn + arr.itemsize
  { arr with data := data }

-- Since the DimIndex is independent of the dtype size, we need to recompute the strides
-- TODO: Would be better to not recompute this over and over. We should find a place to store
-- the 1-based default strides
def getDimIndex [Element a] (x : Tensor) (index : DimIndex) : Err a :=
  let offset := Shape.dimIndexToOffset x.unitStrides index
  let posn := x.startIndex + offset
  if posn < 0 then .error s!"Illegal position: {posn}"
  else getPosition x posn.toNat

def setDimIndex [Element a] (x : Tensor) (index : DimIndex) (v : a): Err Tensor :=
  let offset := Shape.dimIndexToOffset x.unitStrides index
  let posn := x.startIndex + offset
  if posn < 0 then .error s!"Illegal position: {posn}"
  else setPosition x posn.toNat v

-- TODO: remove `Err` by proving all indices are within range
def toList (a : Type) [Tensor.Element a] (x : Tensor) : Err (List a) :=
  let traverseFn ind : Err a := getDimIndex x ind
  x.shape.allDimIndices.traverse traverseFn

def toList! (a : Type) [Tensor.Element a] (x : Tensor) : List a := match toList a x with
| .error _ => []
| .ok x => x

instance BV8Native : Element BV8 where
  dtype := Dtype.mk .uint8 .oneByte
  itemsize := 1
  ofNat n := n
  toByteArray (x : BV8) : ByteArray := x.toByteArray
  fromByteArray arr startIndex := ByteArray.toBV8 arr startIndex

instance Int8Native : Element Int8 where
  dtype := Dtype.mk .int8 .oneByte
  itemsize := 1
  ofNat n := n.toInt8
  toByteArray (x : Int8) : ByteArray := [x.toUInt8].toByteArray
  fromByteArray arr startIndex := (ByteArray.toBV8 arr startIndex).map fun b => Int8.mk b.toUInt8

#guard Int8Native.fromByteArray (Int8Native.toByteArray (-5)) 0 == .ok (-5)

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
#guard toList! BV16 (Tensor.Element.arange BV16 10) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

end Element

namespace Format
open Std.Format

-- Useful for small arrays, e.g. to help with printing and such
-- There are some natural invariants we could check, such as that the
-- trees in a node all have the same height, but since this is just a
-- utility structure we'll keep it simple
inductive Tree a where
| root (xs: List a)
| node (xs: List (Tree a))
deriving BEq, Repr, Inhabited

private def toTree {a : Type} (x : List a) (strides : Strides) : Err (Tree a) :=
  if strides.any fun x => x <= 0 then .error "strides need to be positive" else
  match strides with
  | [] => if x.length == 1 then .ok (.root x) else .error "empty shape that's not an array scalar"
  | [1] => .ok (.root x)
  | [_] => .error "not a unit stride"
  | stride :: strides => do
    let chunks := x.toChunks stride.toNat
    let res <- chunks.traverse (fun x => toTree x strides)
    return .node res

private def toTree! {a : Type} (x : List a) (strides : Strides) : Tree a := match toTree x strides with
| .error _ => .root []
| .ok t => t

#guard toTree! (Element.toList! BV16 (Element.arange BV16 10)) [5, 1] == .node [.root [0, 1, 2, 3, 4], .root [5, 6, 7, 8, 9]]

private def formatRoot [Repr a] (xs : List a) : Std.Format := sbracket (joinSep (List.map repr xs) (text ", "))

private def formatTree1 [Repr a] (shape : List Nat) (t : Tree a) : Err Std.Format :=
  match shape, t with
  | [], .root [x] => .ok $ repr x
  | [n], .root r => if r.length != n then .error "shape mismatch" else .ok (formatRoot r)
  | n :: shape, .node ts => do
    let fmts <- ts.traverse (formatTree1 shape)
    if fmts.length != n then .error "head mismatch" else
    let indented := join (fmts.intersperse (", " ++ line))
    .ok (group (nest 2 ("[" ++ indented ++ "]")))
  | _, _ => .error "format mismatch"

/- This needs some improvement. For example, I'm not able to get the indent to stick
at the end of the "array("

$ bin/tensorlib format 20 2
Got shape [20, 2]
array([[0x0000#16, 0x0001#16],
  [0x0002#16, 0x0003#16],
  [0x0004#16, 0x0005#16],
  ...
-/
private def formatTree [Repr a] (t : Tree a) (shape : Shape) : Err Std.Format := do
  let r <- formatTree1 shape.val t
  return join ["array(", r, ")"]

end Format

def toTree (a : Type) [Repr a] [Element a] (x : Tensor) : Err (Format.Tree a) := do
  let xs <- Element.toList a x
  Format.toTree xs x.unitStrides

def format (a : Type) [Repr a] [Element a] (x : Tensor) : Err Std.Format := do
  let t <- toTree a x
  let f <- Format.formatTree t x.shape
  return f

def str (a : Type) [Repr a] [Tensor.Element a] (x : Tensor) : String := match format a x with
| .error err => s!"Error: {err}"
| .ok s => Std.Format.pretty s 120

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
  let shape := arr.header.shape
  let data := dataOfNpy arr
  let startIndex := 0
  return { dtype, shape, data, startIndex }

private def dtypeToNpy (dtype : Dtype) : Npy.Dtype :=
  let order := match dtype.order with
  | .bigEndian => .bigEndian
  | .littleEndian => .littleEndian
  | .oneByte => .notApplicable
  Npy.Dtype.mk dtype.name order

/-
If we have a non-trivial view, we will need a copy, since strides
and start positions are not included in the .npy file format
-/
private def toNpy (arr : Tensor) : Npy.Ndarray :=
  let arr := if arr.isTriviallyReshapable then arr else arr.copy
  let descr := dtypeToNpy arr.dtype
  let shape := arr.shape
  let header : Npy.Header := { descr, shape }
  let data := arr.data
  let startIndex := 0
  { header, data, startIndex }

section Test

#guard str BV8 (Element.arrayScalar (5 : BV8)) == "array(0x05#8)"
#guard str BV8 (Element.arange BV8 10) == "array([0x00#8, 0x01#8, 0x02#8, 0x03#8, 0x04#8, 0x05#8, 0x06#8, 0x07#8, 0x08#8, 0x09#8])"

private def arr0 := Element.arange BV8 8
#guard get! ((arr0.copyAndReshape! $ Shape.mk [2, 4]).toTree BV8) == .node [.root [0, 1, 2, 3], .root [4, 5, 6, 7]]
#guard get! ((arr0.copyAndReshape! $ Shape.mk [1, 2, 4]).toTree BV8) == .node [.node [.root [0, 1, 2, 3], .root [4, 5, 6, 7]]]

private def arr1 := Element.arange BV8 12
#guard get! ((arr1.copyAndReshape! $ Shape.mk [2, 6]).toTree BV8) == .node [.root [0, 1, 2, 3, 4, 5], .root [6, 7, 8, 9, 10, 11]]
#guard get! ((arr1.copyAndReshape! $ Shape.mk [2, 3, 2]).toTree BV8) == .node [.node [.root [0, 1], .root [2, 3], .root [4, 5]], .node [.root [6, 7], .root [8, 9], .root [10, 11]]]

#guard (zeros (Dtype.float64) $ Shape.mk [2, 2]).nbytes == 2 * 2 * 8
#guard (zeros (Dtype.float64) $ Shape.mk [2, 2]).data.toList.count 0 == 2 * 2 * 8
#guard (ones (Dtype.float64) $ Shape.mk [2, 2]).nbytes == 2 * 2 * 8
#guard (ones (Dtype.float64) $ Shape.mk [2, 2]).data.toList.count 1 == 2 * 2

#guard get! ((Element.ofList Element.BV8Native [1, 2, 3]).toTree BV8) == Format.Tree.root [1, 2, 3]
#guard get! (((Element.ofList Element.BV8Native [0, 1, 2, 3, 4, 5]).reshape! (Shape.mk [2, 3])).toTree BV8) == .node [.root [0, 1, 2], .root [3, 4, 5]]

end Test

end Tensor
end TensorLib
