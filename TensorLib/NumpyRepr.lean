import Mathlib.Tactic

/-!
We largely duplicate the NumPy representation of tensors.

The binary format is described here: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
and here: https://github.com/numpy/numpy/blob/067cb067cb17a20422e51da908920a4fbb3ab851/doc/neps/nep-0001-npy-format.rst

In addition to being efficient, this allows us to directly parse .npy input files.
-/

namespace TensorLib

abbrev Err := Except String

instance [BEq a] : BEq (Err a) where
  beq x y := match x, y with
  | .ok x, .ok y => x == y
  | .error x, .error y => x == y
  | _, _ => false

section Util

-- We generally have huge tensors, so don't show them by default
instance : Repr ByteArray where
  reprPrec x _ :=
    let s := toString x.size
    s!"ByteArray of size {s}"

-- UInt8, UInt64, etc don't work well with bv_decide yet. It's coming
abbrev BV8 := BitVec 8
abbrev BV16 := BitVec 16
abbrev BV32 := BitVec 32
abbrev BV64 := BitVec 64

def BV16ToBytes (n : BV16) : BV8 × BV8 :=
  let n0 := (n >>> 0o00 &&& 0xFF).truncate 8
  let n1 := (n >>> 0o10 &&& 0xFF).truncate 8
  (n0, n1)

def BytesToBV16 (x0 x1 : BV8) : BV16 :=
  (x0.zeroExtend 16 <<< 0o00) |||
  (x1.zeroExtend 16 <<< 0o10)

def BV32ToBytes (n : BV32) : BV8 × BV8 × BV8 × BV8 :=
  let n0 := (n >>> 0o00 &&& 0xFF).truncate 8
  let n1 := (n >>> 0o10 &&& 0xFF).truncate 8
  let n2 := (n >>> 0o20 &&& 0xFF).truncate 8
  let n3 := (n >>> 0o30 &&& 0xFF).truncate 8
  (n0, n1, n2, n3)

def BytesToBV32 (x0 x1 x2 x3 : BV8) : BV32 :=
  (x0.zeroExtend 32 <<< 0o00) |||
  (x1.zeroExtend 32 <<< 0o10) |||
  (x2.zeroExtend 32 <<< 0o20) |||
  (x3.zeroExtend 32 <<< 0o30)

def BV64ToBytes (n : BV64) : BV8 × BV8 × BV8 × BV8 × BV8 × BV8 × BV8 × BV8 :=
  let n0 := (n >>> 0o00 &&& 0xFF).truncate 8
  let n1 := (n >>> 0o10 &&& 0xFF).truncate 8
  let n2 := (n >>> 0o20 &&& 0xFF).truncate 8
  let n3 := (n >>> 0o30 &&& 0xFF).truncate 8
  let n4 := (n >>> 0o40 &&& 0xFF).truncate 8
  let n5 := (n >>> 0o50 &&& 0xFF).truncate 8
  let n6 := (n >>> 0o60 &&& 0xFF).truncate 8
  let n7 := (n >>> 0o70 &&& 0xFF).truncate 8
  (n0, n1, n2, n3, n4, n5, n6, n7)

def BytesToBV64 (x0 x1 x2 x3 x4 x5 x6 x7 : BV8) : BV64 :=
  (x0.zeroExtend 64 <<< 0o00) |||
  (x1.zeroExtend 64 <<< 0o10) |||
  (x2.zeroExtend 64 <<< 0o20) |||
  (x3.zeroExtend 64 <<< 0o30) |||
  (x4.zeroExtend 64 <<< 0o40) |||
  (x5.zeroExtend 64 <<< 0o50) |||
  (x6.zeroExtend 64 <<< 0o60) |||
  (x7.zeroExtend 64 <<< 0o70)

theorem BV64BytesRoundTrip (n : BV64) :
  let (x0, x1, x2, x3, x4, x5, x6, x7) := BV64ToBytes n
  let n' := BytesToBV64 x0 x1 x2 x3 x4 x5 x6 x7
  n = n' := by
    unfold BV64ToBytes BytesToBV64
    bv_decide

theorem BV64BytesRoundTrip1 (x0 x1 x2 x3 x4 x5 x6 x7 : BV8) :
  let n := BytesToBV64 x0 x1 x2 x3 x4 x5 x6 x7
  let (x0', x1', x2', x3', x4', x5', x6', x7') := BV64ToBytes n
  x0 = x0' &&
  x1 = x1' &&
  x2 = x2' &&
  x3 = x3' &&
  x4 = x4' &&
  x5 = x5' &&
  x6 = x6' &&
  x7 = x7' := by
    unfold BV64ToBytes BytesToBV64
    bv_decide

end Util

namespace NumpyDtype

inductive ByteOrder where
| native
| littleEndian
| bigEndian
| notApplicable
deriving BEq, Repr

namespace ByteOrder

def toChar (x : ByteOrder) := match x with
| native => '='
| littleEndian => '<'
| bigEndian => '>'
| notApplicable => '|'

def fromChar (c : Char) : Err ByteOrder := match c with
| '=' => .ok native
| '<' => .ok littleEndian
| '>' => .ok bigEndian
| '|' => .ok notApplicable
| _ => .error s!"can't parse byte order: {c}"

def UInt8ToBV8 (n : UInt8) : BV8 := BitVec.ofFin n.val
def BV8ToUInt8 (n : BV8) : UInt8 := UInt8.ofNat n.toFin

private def getBV16Aux (order : ByteOrder) (x0 x1 : BV8) : Err BV16 :=
  match order with
  | .notApplicable | .native => .error "illegal byte order"
  | .littleEndian => .ok (BytesToBV16 x0 x1)
  | .bigEndian => .ok (BytesToBV16 x1 x0)

def getBV16 (order : ByteOrder) (x : ByteArray) (startIndex : Nat) : Err BV16 :=
  let n := startIndex
  if H7 : n + 1 < x.size then
    let H0 : n + 0 < x.size := by linarith
    let H1 : n + 1 < x.size := by linarith
    let x0 := UInt8ToBV8 $ x.get (Fin.mk _ H0)
    let x1 := UInt8ToBV8 $ x.get (Fin.mk _ H1)
    getBV16Aux order x0 x1
  else .error "Index out of range"

private def getBV32Aux (order : ByteOrder) (x0 x1 x2 x3 : BV8) : Err BV32 :=
  match order with
  | .notApplicable | .native => .error "illegal byte order"
  | .littleEndian => .ok (BytesToBV32 x0 x1 x2 x3)
  | .bigEndian => .ok (BytesToBV32 x3 x2 x1 x0)

def getBV32 (order : ByteOrder) (x : ByteArray) (startIndex : Nat) : Err BV32 :=
  let n := startIndex
  if H7 : n + 3 < x.size then
    let H0 : n + 0 < x.size := by linarith
    let H1 : n + 1 < x.size := by linarith
    let H2 : n + 2 < x.size := by linarith
    let H3 : n + 3 < x.size := by linarith
    let x0 := UInt8ToBV8 $ x.get (Fin.mk _ H0)
    let x1 := UInt8ToBV8 $ x.get (Fin.mk _ H1)
    let x2 := UInt8ToBV8 $ x.get (Fin.mk _ H2)
    let x3 := UInt8ToBV8 $ x.get (Fin.mk _ H3)
    getBV32Aux order x0 x1 x2 x3
  else .error "Index out of range"

private def getBV64Aux (order : ByteOrder) (x0 x1 x2 x3 x4 x5 x6 x7 : BV8) : Err BV64 :=
  match order with
  | .notApplicable | .native => .error "illegal byte order"
  | .littleEndian => .ok (BytesToBV64 x0 x1 x2 x3 x4 x5 x6 x7)
  | .bigEndian => .ok (BytesToBV64 x7 x6 x5 x4 x3 x2 x1 x0)

def getBV64 (order : ByteOrder) (x : ByteArray) (startIndex : Nat) : Err BV64 :=
  let n := startIndex
  if H7 : n + 7 < x.size then
    let H0 : n + 0 < x.size := by linarith
    let H1 : n + 1 < x.size := by linarith
    let H2 : n + 2 < x.size := by linarith
    let H3 : n + 3 < x.size := by linarith
    let H4 : n + 4 < x.size := by linarith
    let H5 : n + 5 < x.size := by linarith
    let H6 : n + 6 < x.size := by linarith
    let x0 := UInt8ToBV8 $ x.get (Fin.mk _ H0)
    let x1 := UInt8ToBV8 $ x.get (Fin.mk _ H1)
    let x2 := UInt8ToBV8 $ x.get (Fin.mk _ H2)
    let x3 := UInt8ToBV8 $ x.get (Fin.mk _ H3)
    let x4 := UInt8ToBV8 $ x.get (Fin.mk _ H4)
    let x5 := UInt8ToBV8 $ x.get (Fin.mk _ H5)
    let x6 := UInt8ToBV8 $ x.get (Fin.mk _ H6)
    let x7 := UInt8ToBV8 $ x.get (Fin.mk _ H7)
    getBV64Aux order x0 x1 x2 x3 x4 x5 x6 x7
  else .error "Index out of range"

-- #guard doesn't work here. Not sure why.
#eval
  let n : BV64 := 0x3FFAB851EB851EB8
  let (x0, x1, x2, x3, x4, x5, x6, x7) := BV64ToBytes n
  -- Big-endian array layout
  let data := ByteArray.mk (Array.mkArray8 (BV8ToUInt8 x0) (BV8ToUInt8 x1) (BV8ToUInt8 x2) (BV8ToUInt8 x3) (BV8ToUInt8 x4) (BV8ToUInt8 x5) (BV8ToUInt8 x6) (BV8ToUInt8 x7))
  do
    let n' <- getBV64 ByteOrder.littleEndian data 0
    return n == n'

end ByteOrder

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
deriving BEq, Repr

namespace Name

instance : ToString Name where
  toString x := (repr x).pretty

--! Number of bytes used by each element of the given dtype
def numBytes (x: Name): Nat := match x with
| float64 | int64 | uint64 => 8
| float32 | int32 | uint32 => 4
| float16 | int16 | uint16 => 2
| bool | int8 | uint8 => 1

/-!
Parse a numpy dtype.

Disk formats found through experimentation. Not sure why there are
both '<' and '|' as prefixes. The first character represents the
byte order: https://numpy.org/doc/2.1/reference/generated/numpy.dtype.byteorder.html
'<' is little endian, '|' is "not applicable" Not sure why bool showed up
as both.
-/
def fromString (s : String) : Err Name := match s with
| "b1" => .ok bool
| "i1" => .ok int8
| "i2" => .ok int16
| "i4" => .ok int32
| "i8" => .ok int64
| "u1" => .ok uint8
| "u2" => .ok uint16
| "u4" => .ok uint32
| "u8" => .ok uint64
| "f2" => .ok float16
| "f4" => .ok float32
| "f8" => .ok float64
| _ => .error s!"Can't parse {s} as a dtype"

def toString (t : Name) : String := match t with
| bool => "b1"
| int8 => "i1"
| int16 => "i2"
| int32 => "i4"
| int64 => "i8"
| uint8 => "u1"
| uint16 => "u2"
| uint32 => "u3"
| uint64 => "u4"
| float16 => "f2"
| float32 => "f4"
| float64 => "f8"

end Name

end NumpyDtype -- temporarily close the namespace so we don't duplicate the structure name

structure NumpyDtype where
  name : NumpyDtype.Name
  order : NumpyDtype.ByteOrder
deriving BEq, Repr

namespace NumpyDtype

def numBytes (t : NumpyDtype) := t.name.numBytes

def fromString (s : String) : Err NumpyDtype :=
  if s.length == 0 then .error "Empty dtype string" else
  do
    let order <- ByteOrder.fromChar (s.get 0)
    let name <- Name.fromString (s.drop 1)
    return { name, order }

def toString (t : NumpyDtype) : String := t.order.toChar.toString.append t.name.toString

-- Examples
private def littleEndian (name : Name) : NumpyDtype := { name, order := ByteOrder.littleEndian }
private def uint8 : NumpyDtype := { name := Name.uint8 , order := ByteOrder.notApplicable }
private def uint16 : NumpyDtype := littleEndian Name.uint16
private def uint32 : NumpyDtype := littleEndian Name.uint32
private def uint64 : NumpyDtype := littleEndian Name.uint64

end NumpyDtype

/-!
Shapes and strides in tensors are represented as lists, where the length of the
list is the number of dimensions. For example, a 2 x 3 matrix has a shape of [2, 3].

```
>>> x = np.arange(6).reshape(2, 3)
>>> x
array([[0, 1, 2],
       [3, 4, 5]])
>>> x.shape
(2, 3)
```

(See https://web.mit.edu/dvp/Public/numpybook.pdf for extensive discussion of shape and stride.)

What about the unit element? What is the shape of an empty tensor? For example,
what is the shape of the 1D empty array `[]`? We follow NumPy by defining the shape
as a 1d matrix with 0 elements.

```
>>> np.array([]).shape
(0,)
```

(Assuming we allow 0s in other dimensions, we can shape-check the empty tensor at other shapes, e.g.
`np.array([]).reshape([1,2,3,0,5])` succeeds.)

The only way to have an empty shape in Numpy is as a "scalar array" (or "array scalar" depending on the document.)

    https://numpy.org/doc/stable/reference/arrays.scalars.html

A scalar array is conceptually a boxed scalar with an empty shape

```
>>> np.array(1).shape
()
```

`None` also yields an empty shape, but we ignore this case.

```
>>> np.array(None).shape
()
```

Strides also are empty for scalar arrays.

```
>>> np.array(1).strides
()
```
-/
abbrev Shape := List Nat
abbrev Strides := List Nat
abbrev Index := List Nat

namespace Shape

--! The number of elements in a tensor. All that's needed is the shape for this calculation.
def count (s : Shape) : Nat := s.prod

/-!
Strides can be computed from the shape by figuring out how many elements you
need to jump over to get to the next spot and mulitplying by the bytes in each
element.

A given shape can have different strides if the tensor is a view of another
tensor. For example, in a square matrix, the transposed matrix view has the same
shape but the strides change.

Broadcasting does funny things to strides, e.g. the stride can be 0 on a dimension,
so this is just the default case.
-/
def defaultStrides (dtype : NumpyDtype) (s : Shape) : Strides :=
  if H : s.isEmpty then [] else
  let s' := s.reverse
  let bytes := dtype.numBytes
  let rec loop (xs : List ℕ) (lastShape lastDimSize : ℕ): List ℕ := match xs with
  | [] => []
  | d :: ds =>
    let rest := loop ds (lastShape * lastDimSize) d
    lastShape * lastDimSize :: rest
  let ok : s' ≠ [] := by
    have H1 : s' = s.reverse := by trivial
    simp [H1]
    simp at H
    exact H
  let res : Strides := dtype.numBytes :: loop s'.tail bytes (s'.head ok)
  res.reverse

#guard defaultStrides NumpyDtype.uint32 [2] == [4]
#guard defaultStrides NumpyDtype.uint32 [2, 3] == [12, 4]
#guard defaultStrides NumpyDtype.uint32 [2, 3, 5, 7] == [420, 140, 28, 4]

end Shape

/-!
This is the header of the on-disk Numpy format, typically with the .npy file extension.

https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#format-version-1-0
-/
structure NumpyHeader where
  -- Version is major.minor
  major : UInt8
  minor : UInt8
  -- Nat is definitely not the final representation. We need
  -- some function from python type descriptor and element to the appropriate
  -- Lean type.
  descr : NumpyDtype
  -- Fortran order is column-major
  -- C order is row major
  -- We can transpose by flipping this bit
  fortranOrder : Bool
  shape : Shape
  deriving Repr

namespace NumpyHeader

/-!
A npy binary file has a header, some padding, then the data. This method computes the
size of the data portion of the file.
-/
def dataSize (header : NumpyHeader): Nat := header.descr.numBytes * header.shape.count

end NumpyHeader


/-!
A NumpyRepr is the data of the tensor, along with all metadata required to do
efficient computation. The `startIndex`, `endIndex`, and `strides` are inferred
from parsing or computed during a view creation.
-/
structure NumpyRepr where
  header : NumpyHeader
  data : ByteArray
  startIndex : Nat -- Pointer to the first byte of ndarray data, called `offset` in numpy
  endIndex : Nat   -- Pointer to the byte after the last byte of ndarray data
  -- invariant: startIndex <= endIndex
  -- invariant: endIndex - startIndex = header.descr.numBytes * shape.count
  -- invariant: header.endIndex <= |data|
  -- invariant: header.dataSize <= |data|
  strides : Strides
  deriving Repr

namespace NumpyRepr

--! shape
def shape (x : NumpyRepr) : Shape := x.header.shape

--! number of dimensions
def ndim (x : NumpyRepr) : ℕ := x.shape.length

--! number of elements
def size (x : NumpyRepr) : ℕ := x.shape.count

--! number of bytes representing each element
def itemsize (x : NumpyRepr) : ℕ := x.header.descr.numBytes

--! number of bytes representing the entire tensor
def nbytes (x : NumpyRepr) : ℕ := x.itemsize * x.size

/-!
Reshaping a tensor is just re-interpreting the elements in
a different order. This will also impact the iteration order/strides
when we implement those.
-/
def reshape (x : NumpyRepr) (shape : Shape) : Err NumpyRepr := do
  if x.shape.count == shape.count then
    return { x with header.shape := shape }
  else
    .error "Reshaping must have the same number of implied elements"

class TensorElement (a: Type) where
  numBytes : Nat
  fromByteArray (byteOrder : NumpyDtype.ByteOrder) (arr : ByteArray) (startIndex : Nat) : Err a

namespace TensorElement

-- Style of repeating the arguments to a class followed by variables was
-- suggested by Lean office hours folks
-- variable (a: Type)
-- variable [Add a][Sub a][Mul a][Neg a]

-- ...Lots of definitions here using the variables above...

instance BV16 : TensorElement BV16 where
  numBytes := 2
  fromByteArray byteOrder arr startIndex :=
    if arr.size < startIndex + 2 -- would like to use `numBytes` but it's not resolving
    then .error "Wrong size array"
    else byteOrder.getBV16 arr startIndex

instance BV32: TensorElement BV32 where
  numBytes := 4
  fromByteArray byteOrder arr startIndex :=
    if arr.size < startIndex + 4
    then .error "Wrong size array"
    else byteOrder.getBV32 arr startIndex

instance BV64: TensorElement BV64 where
  numBytes := 8
  fromByteArray byteOrder arr startIndex :=
    if arr.size < startIndex + 8
    then .error "Wrong size array"
    else byteOrder.getBV64 arr startIndex

end TensorElement


namespace Index

-- Indexing utility
private def start (strides : Strides) (index : Index) : Err ℕ :=
  match strides, index with
  | [], [] => .error "Can not index with empty list"
  | [], _ :: _ | _ :: _, [] => .error "Unequal lengths: strides and index"
  | stride :: strides, i :: index => (start strides index).map (. + i * stride)

-- Get a single value from the tensor.
-- TODO: Replace get! with the Fin version. I tried this for a couple hours
-- and failed.
private def getBytes! (x : NumpyRepr) (index : Index) : Err (List UInt8) := do
  let gap <- start x.strides index
  let i := x.startIndex + gap
  let rec loop (n : ℕ) (acc : List UInt8) : List UInt8 :=
    match n with
    | 0 => acc
    | n + 1 => loop n (x.data.get! (i + n) :: acc)
  return loop x.itemsize []

-- This is a blind index into the array, disregarding the shape.
def rawIndex (typ : TensorElement a) (x : NumpyRepr) (index : Nat) : Err a :=
  if typ.numBytes != x.itemsize then .error "byte size mismatch" else -- TODO: Lift this check out so we only do it once
  typ.fromByteArray x.header.descr.order x.data (x.startIndex + (index * typ.numBytes))

end Index

def transpose (x : NumpyRepr) : NumpyRepr :=
  let shape := x.header.shape.reverse
  let header := { x.header with shape }
  let strides := x.strides.reverse
  { x with header, strides }

/-!
Broadcasting is a convenience and performance trick to allow operations that expect the same
shaped arguments to work on non-matching arguments.  For example, we would like to be able
to add 1 to each element of a tensor without building the all-1s tensor in memory.
It involves applying the following rules to two tensors

1. If the shape of one is smaller than the other, pad the smaller one
   with 1s until they are the same length
2. For each pair of numbers at each index, to broadcast either the
   numbers must be the same, or one of them should be 1. In the later
   case we replace that shape with the other number

For example, we broadcast (3, 2, 1) (2, 7) to (3, 2, 7).

A: (3, 2, 1)
B: (2, 7)

Rule 1

A: (3, 2, 1)
B: (1, 2, 7)

Rule 2

A: (3, 2, 1)
B: (3, 2, 7)

Rule 2

A: (3, 2, 7)
B: (3, 2, 7)
-/
structure Broadcast where
  left : Shape
  right : Shape
  deriving BEq, Repr

section Broadcast

-- In broadcasting, we first extend the shorter array by prefixing 1s.
-- NKI semantics currently suffixes 1s in some cases, so be explicit about
-- the naming.
private def oneExtendPrefix (b : Broadcast) : Broadcast :=
  let n1 := b.left.length
  let n2 := b.right.length
  if n1 <= n2
  then { b with left := List.replicate (n2 - n1) 1 ++ b.left }
  else { b with right := List.replicate (n1 - n2) 1 ++ b.right }

private theorem oneExtendPrefixLength (b : Broadcast) :
  let b' := oneExtendPrefix b
  b'.left.length = b'.right.length := by
  cases b
  rename_i left right
  simp [oneExtendPrefix]
  by_cases H : left.length <= right.length
  . simp [H]
  . simp [H]
    rw [Nat.sub_add_cancel]
    linarith

private def matchPairs (b : Broadcast) : Option Shape :=
  if b.left.length != b.right.length then none else
  let f xy := match xy with
    | (x, y) =>
      if x == y then some x
      else if x == 1 then some y
      else if y == 1 then some x
      else none
  traverse f (b.left.zip b.right)

--! Returns the shape resulting from broadcast the arguments
def broadcast (b : Broadcast) : Option Shape := matchPairs (oneExtendPrefix b)

--! Whether broadcasting is possible
def canBroadcast (b : Broadcast) : Bool := (broadcast b).isSome

#guard matchPairs (Broadcast.mk [1, 2, 3] [7, 2, 1]) == some [7, 2, 3]
#guard broadcast (Broadcast.mk [1, 2, 3] [7, 7, 9, 2, 1]) == some [7, 7, 9, 2, 3]

-- todo: add plausible properties when property-based testing settles down in Lean-land
#guard
 let x1 := [1,2,3]
 let x2 := [2,3]
 let b1 := Broadcast.mk x1 x1
 let b2 := Broadcast.mk x1 x2
 oneExtendPrefix b1 == b1 &&
 broadcast b2 == broadcast b1 &&
 broadcast b2 == some [1, 2, 3]

end Broadcast

end NumpyRepr
end TensorLib
