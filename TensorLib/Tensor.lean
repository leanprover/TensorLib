/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/

import Batteries.Data.List -- for `toChunks`
import TensorLib.Broadcast
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
  if arr.shape.count != shape.count then .error s!"Incompatible shapes: {arr.shape} {shape}" else
  let arr := arr.copy
  .ok { arr with shape, unitStrides := shape.unitStrides }

def copyAndReshape! (arr : Tensor) (shape : Shape) : Tensor :=
  get! (copyAndReshape arr shape)

def reshape (arr : Tensor) (shape : Shape) : Err Tensor :=
  if arr.shape == shape then .ok arr else
  if arr.shape.count != shape.count then .error s!"Incompatible shapes: {arr.shape} {shape}" else
  if arr.isTriviallyReshapable
  then .ok { arr with shape, unitStrides := shape.unitStrides }
  else copyAndReshape arr shape

def reshape! (t : Tensor) (s : Shape) : Tensor := get! $ t.reshape s

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

-- The result shape is always equal to `toShape` so we don't need to remember it
private def broadcastStrides (fromShapeAndStrides : List (Nat × Int)) (toShape : Shape) : Err Strides :=
  let rec loop (fromShapeAndStrides : List (Nat × Int)) (toShape : List Nat) : Err Strides :=
    match fromShapeAndStrides, toShape with
    | [], [] => .ok []
    | (xDim, xStride) :: xs, yDim :: ys =>
      if xDim == yDim then do
        let rest <- loop xs ys
        return xStride :: rest
      else if xDim == 1 then do
        let rest <- loop xs ys
        return 0 :: rest
      else .error "Can't broadcast dimension"
    | [], _ :: ys => do
      let rest <- loop [] ys
      return 0 :: rest
    | _ :: _, [] => .error "Can't broadcast dimension"
  (loop fromShapeAndStrides.reverse toShape.val.reverse).map fun x => x.reverse

#guard broadcastStrides [] (Shape.mk []) == .ok []
#guard broadcastStrides [(5, 8)] (Shape.mk [5]) == .ok [8]
#guard broadcastStrides [(5, 8)] (Shape.mk [10, 5]) == .ok [0, 8]
#guard broadcastStrides [(5, 8)] (Shape.mk [15, 10, 5]) == .ok [0, 0, 8]
#guard !(broadcastStrides [(1, 0)] (Shape.mk [])).isOk
#guard !(broadcastStrides [(1, 0), (2, 2), (3, 8)] (Shape.mk [2, 3])).isOk
#guard !(broadcastStrides [(1, 0)] (Shape.mk [])).isOk
#guard !(broadcastStrides [(2, 0), (2, 2), (3, 8)] (Shape.mk [1, 2, 3])).isOk

/-
Theorems to prove:
* arr.broadcastTo arr.shape == arr
* (arr.broadcastTo s1).broadcastTo s2 == arr.broadcastTo s2
* ...
-/
def broadcastTo (arr : Tensor) (shape : Shape) : Err Tensor :=
  match Broadcast.broadcast { left := arr.shape, right := shape } with
  | none => .error s!"Can't broadcast {arr.shape} to {shape}"
  | some shape' =>
    if shape != shape' then .error s!"Can't broadcast {arr.shape} to {shape}" else do
    let strides <- broadcastStrides (arr.shape.val.zip arr.strides) shape
    .ok $ Tensor.mk arr.dtype shape arr.data arr.startIndex strides

def broadcastTo! (arr : Tensor) (shape : Shape) : Tensor := get! $ broadcastTo arr shape

def broadcast (arr1 : Tensor) (arr2 : Tensor) : Err (Tensor × Tensor) :=
  match Broadcast.broadcast { left := arr1.shape, right := arr2.shape } with
  | none => .error "Can't broadcast"
  | some shape => do
  let arr1 <- arr1.broadcastTo shape
  let arr2 <- arr2.broadcastTo shape
  return (arr1, arr2)

def arrayScalar (dtype : Dtype) (arr : ByteArray) : Err Tensor :=
  if dtype.itemsize != arr.size then .error "data size mismatch" else
  .ok { dtype := dtype, shape := Shape.empty, data := arr }

def arrayScalarNat (dtype : Dtype) (n : Nat) : Err Tensor := do
  let arr <- dtype.byteArrayOfNat n
  arrayScalar dtype arr

def arrayScalarNat! (dtype : Dtype) (n : Nat) : Tensor := get! $ arrayScalarNat dtype n

def arrayScalarInt (dtype : Dtype) (n : Int) : Err Tensor := do
  let arr <- dtype.byteArrayOfInt n
  arrayScalar dtype arr

def arrayScalarInt! (dtype : Dtype) (n : Int) : Tensor := get! $ arrayScalarInt dtype n

def arange (dtype : Dtype) (n : Nat) : Err Tensor := do
  let size := dtype.itemsize
  let mut data := ByteArray.mkEmpty (n * size)
  for i in [0:n] do
    let bytes <- dtype.byteArrayOfNat i
    data := ByteArray.copySlice bytes 0 data (i * size) size
  return { dtype := dtype, shape := Shape.mk [n], data }

def arange! (dtype : Dtype) (n : Nat) : Tensor := get! $ arange dtype n

-- This is a blind index into the array, disregarding the shape.
def getPosition (arr : Tensor) (position : Nat) : ByteArray :=
  arr.data.copySlice (position * arr.itemsize) (ByteArray.mkEmpty arr.itemsize) 0 arr.itemsize

#guard
  let tp :=  Dtype.uint32
  let arr := arange! tp 4
  let v := getPosition arr 3
  v.toList == [3, 0, 0, 0]

-- This is a blind index into the array, disregarding the shape.
def setPosition (arr : Tensor) (n : Nat) (v : ByteArray): Tensor :=
  let size := arr.itemsize
  let posn := n * size
  { arr with data := v.copySlice 0 arr.data posn size }

#guard
  let tp :=  Dtype.uint8
  let arr := arange! tp 4
  let arr := setPosition arr 0 (tp.byteArrayOfNat! 7)
  arr.data.data == #[7, 1, 2, 3]

/-
We now define some constructors for Tensors that are mostly useful
for testing. Instead of requiring the `dtype` argument, it may be
better to use a class, e.g.

    class DtypeOf a where
      dtype : Dtype

and let `ofList` infer the dtype. The reason we're not going with
this now is that there is no obvious canonical candidate for the
dtype. E.g. for Nat, we could reasonably want uint8 for small examples,
uint16 for bigger ones, etc.
-/
def ofNatList (dtype : Dtype) (ns : List Nat) : Err Tensor := do
  if !dtype.isUint then .error "not a uint type" else
  let size := dtype.itemsize
  let arr := Tensor.zeros dtype (Shape.mk [ns.length])
  let mut data := arr.data
  let mut posn := 0
  for n in ns do
    let v <- dtype.byteArrayOfNat n
    data := v.copySlice 0 data posn size
    posn := posn + size
  .ok { arr with data := data }

def ofNatList! (dtype : Dtype) (ns : List Nat) : Tensor := get! $ ofNatList dtype ns

def ofIntList (dtype : Dtype) (ns : List Int) : Err Tensor := do
  if !dtype.isInt then .error "not an int type" else
  let size := dtype.itemsize
  let arr := Tensor.zeros dtype (Shape.mk [ns.length])
  let mut data := arr.data
  let mut posn := 0
  for n in ns do
    let v <- dtype.byteArrayOfInt n
    data := v.copySlice 0 data posn size
    posn := posn + size
  .ok { arr with data := data }

def ofIntList! (dtype : Dtype) (ns : List Int) : Tensor := get! $ ofIntList dtype ns

def getDimIndex (arr : Tensor) (index : DimIndex) : Err ByteArray :=
  if arr.shape.ndim != index.length then .error "getDimIndex: index mismatch" else
  let offset := Shape.dimIndexToOffset arr.unitStrides index
  let posn := arr.startIndex + offset
  if posn < 0 then .error s!"Illegal position: {posn}" else
  let res := getPosition arr posn.toNat
  .ok res

def getDimIndex! (arr : Tensor) (index : DimIndex) : ByteArray := get! $ getDimIndex arr index

#guard
  let arr := arrayScalarNat! Dtype.uint8 25
  let v := getDimIndex! arr []
  v.data == #[25]

def setDimIndex (arr : Tensor) (index : DimIndex) (v : ByteArray) : Err Tensor :=
  let offset := Shape.dimIndexToOffset arr.unitStrides index
  let posn := arr.startIndex + offset
  if posn < 0 then .error s!"Illegal position: {posn}"
  else .ok $ setPosition arr posn.toNat v

def toList (arr : Tensor) : Err (List ByteArray) :=
  arr.shape.allDimIndices.mapM (getDimIndex arr)

/-
Similar to np.array_equal, but requires the dtype to be the same
-/
def arrayEqual (x y : Tensor) : Bool :=
  x.dtype == y.dtype && x.shape == y.shape && match x.toList, y.toList with
  | .error _, _ | _, .error _ => false
  | .ok xs, .ok ys => xs.length == ys.length && (xs.zip ys).all fun (x, y) => x == y

/-
Like NumPy's `astype`: https://numpy.org/doc/2.1/reference/generated/numpy.ndarray.astype.html
Afaict, NumPy never fails due to overflow/underflow during type conversions, so use the "overflow"
variant of type casting.

`astype` will make a copy of the tensor iff `toDtype != arr.dtype`.
-/
def astype (arr : Tensor) (toDtype : Dtype) : Err Tensor := do
  if arr.dtype == toDtype then .ok arr else
  let mut res : Tensor := {
    dtype := toDtype,
    shape := arr.shape,
    data := ByteArray.mkEmpty (arr.size * toDtype.itemsize)
  }
  let iter := DimsIter.make arr.shape
  for dimIndex in iter do
    let v <- arr.getDimIndex dimIndex
    let v' := Dtype.castOverflow arr.dtype v toDtype
    let res' <- res.setDimIndex dimIndex v'
    res := res'
  return res

def astype! (arr : Tensor) (toDtype : Dtype) : Tensor := get! $ astype arr toDtype

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

namespace Tree

-- Traverse the left-most branch to infer the shape. Don't bother checking that it's uniform
-- since presumably it was created by a `arr.toTree` variant.
private def inferShape (t : Tree a) : List Nat := match t with
| .root xs => [xs.length]
| .node [] => impossible
| .node (t :: ts) => (1 + ts.length) :: inferShape t

private def mapM [Monad m] (f : a -> m b) (t : Tree a) : m (Tree b) :=
  map1 f t
where
  map1 f
  | .root xs => do
    let xs' <- xs.mapM f
    return .root xs'
  | .node ts => do
    let ts' <- mapN f ts
    return .node ts'
  mapN f
  | [] => return []
  | t :: ts => do
    let t' <- map1 f t
    let ts' <- mapN f ts
    return t' :: ts'

private def map (f : a -> Id b) (t : Tree a) : Tree b := mapM f t

private def formatRoot [Repr a] (xs : List a) : Lean.Format :=
  sbracket (joinSep (List.map repr xs) (text ", "))

private def formatTree1 [Repr a] (t : Tree a) (shape : List Nat) : Err Std.Format :=
  match shape, t with
  | [], .root [x] => .ok $ repr x
  | [n], .root r => if r.length != n then .error "shape mismatch" else .ok (formatRoot r)
  | n :: shape, .node ts => do
    let fmts <- ts.traverse (fun t => formatTree1 t shape)
    if fmts.length != n then .error "head mismatch" else
    let indented := join (fmts.intersperse (", " ++ line))
    .ok (group (nest 2 ("[" ++ indented ++ "]")))
  | _, _ => .error "format mismatch"

def format [Repr a] (t : Tree a) : Err Lean.Format := do
  let r <- formatTree1 t t.inferShape
  return join ["array(", r, ")"]

def format! [Repr a] (t : Tree a) : Lean.Format := get! $ format t

end Tree

private def listToTree (arr : List a) (strides : Strides) : Err (Tree a) :=
  if strides.any fun x => x <= 0 then .error "strides need to be positive" else
  match strides with
  | [] => if arr.length == 1 then .ok (.root arr) else .error "empty shape that's not an array scalar"
  | [1] => .ok (.root arr)
  | [_] => .error "not a unit stride"
  | stride :: strides => do
    let chunks := arr.toChunks stride.toNat
    let res <- chunks.mapM (fun x => listToTree x strides)
    return .node res


/- This needs some improvement. For example, I'm not able to get the indent to stick
at the end of the "array("

$ bin/tensorlib format 20 2
Got shape [20, 2]
array([[0x0000#16, 0x0001#16],
  [0x0002#16, 0x0003#16],
  [0x0004#16, 0x0005#16],
  ...
-/

end Format

def toByteArrayTree (arr : Tensor) : Err (Format.Tree ByteArray) := do
  let xs <- arr.toList
  -- Now that we have the elements in a list, we don't care about the strides `arr` which
  -- could have been complex (e.g. negative). Now we just want standard unit strides over the list
  Format.listToTree xs arr.shape.unitStrides

def toIntTree (arr : Tensor) : Err (Format.Tree Int) := do
  let t <- arr.toByteArrayTree
  t.mapM arr.dtype.byteArrayToInt

def toIntTree! (arr : Tensor) : Format.Tree Int := get! $ toIntTree arr

def toNatTree (arr : Tensor) : Err (Format.Tree Nat) := do
  let t <- arr.toByteArrayTree
  t.mapM arr.dtype.byteArrayToNat

def toNatTree! (arr : Tensor) : Format.Tree Nat := get! $ toNatTree arr

def formatInt (arr : Tensor) : Err Std.Format := do
  let t <- arr.toIntTree
  t.format

def formatNat (arr : Tensor) : Err Std.Format := do
  let t <- arr.toNatTree
  t.format

private def dataOfNpy (arr : Npy.Ndarray) : ByteArray :=
  let dst := ByteArray.mkEmpty arr.nbytes
  arr.data.copySlice arr.startIndex dst 0 arr.nbytes

/-
Makes a copy of the data, dropping the header and padding.
Probably not a great choice, but sticking with it for now.
I want to avoid writing .npy files with wrong header data.
-/
def ofNpy (arr : Npy.Ndarray) : Err Tensor := do
  match arr.order.toByteOrder with
  | .none => .error "can't convert byte order"
  | .some order =>
  let dtype <- Dtype.make arr.dtype.name order
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

open TensorLib.Tensor.Format.Tree

#guard
  let arr := arrayScalarNat! Dtype.uint8 5
  let t := arr.toNatTree!
  t == .root [5]

#guard
  let arr := (arange! Dtype.uint16 10).reshape! (Shape.mk [2, 5])
  let t := arr.toNatTree!
  t == node [root [0, 1, 2, 3, 4], root [5, 6, 7, 8, 9]]

#guard (zeros Dtype.float64 $ Shape.mk [2, 2]).nbytes == 2 * 2 * 8
#guard (zeros Dtype.float64 $ Shape.mk [2, 2]).data.toList.count 0 == 2 * 2 * 8
#guard (ones Dtype.float64 $ Shape.mk [2, 2]).nbytes == 2 * 2 * 8
#guard (ones Dtype.float64 $ Shape.mk [2, 2]).data.toList.count 1 == 2 * 2

#guard
  let t1 := (arange! Dtype.uint8 6).reshape! (Shape.mk [2, 3])
  let t2 := t1.broadcastTo! (Shape.mk [2, 2, 3])
  let tree := t2.toNatTree!
  let n1 := node [ root [0, 1, 2], root [3, 4, 5] ]
  let tree' := node [ n1, n1 ]
  tree == tree'

#guard
  let t1 := (arange! Dtype.uint8 8).reshape! (Shape.mk [2, 1, 1, 4])
  let t2 := t1.broadcastTo! (Shape.mk [2, 3, 3, 4])
  let tree := t2.toNatTree!
  let r1 := root [0, 1, 2, 3]
  let r2 := root [4, 5, 6, 7]
  let n1 := node [ r1, r1, r1 ]
  let n2 := node [ r2, r2, r2 ]
  let tree' := node [ node [ n1, n1, n1 ], node [n2, n2, n2] ]
  tree == tree'

#guard
  let t := (arange! Dtype.uint8 6).reshape! (Shape.mk [2, 3])
  let t1 := t.astype! Dtype.uint16
  let t1 := t1.astype! Dtype.uint8
  Tensor.arrayEqual t t1

#guard
  let t := (arange! Dtype.uint8 6).reshape! (Shape.mk [2, 3])
  let t1 := t.astype! Dtype.uint64
  let t1 := t1.astype! Dtype.uint32
  let t1 := t1.astype! Dtype.uint16
  let t1 := t1.astype! Dtype.uint8
  Tensor.arrayEqual t t1

#guard
  let t := (arange! Dtype.uint8 6).reshape! (Shape.mk [2, 3])
  let t1 := t.astype! Dtype.int8
  let t1 := t1.astype! Dtype.uint32
  let t1 := t1.astype! Dtype.uint16
  let t1 := t1.astype! Dtype.float32
  let t1 := t1.astype! Dtype.float64
  let t1 := t1.astype! Dtype.uint8
  Tensor.arrayEqual t t1

end Test

end Tensor
end TensorLib
