import Mathlib.Tactic

/-!
We largely duplicate the NumPy representation of tensors.

The binary format is described here: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
and here: https://github.com/numpy/numpy/blob/067cb067cb17a20422e51da908920a4fbb3ab851/doc/neps/nep-0001-npy-format.rst

In addition to being an efficient representation, this allows us to directly parse .npy input files.
-/

namespace TensorLib

abbrev Err := Except String

instance [BEq a] : BEq (Err a) where
  beq x y := match x, y with
  | .ok x, .ok y => x == y
  | .error x, .error y => x == y
  | _, _ => false

namespace Util

-- We generally have huge tensors, so don't show them by default
instance ByteArrayRepr : Repr ByteArray where
  reprPrec x _ :=
    let s := toString x.size
    s!"ByteArray of size {s}"

def dot [Add a][Mul a][Zero a] (x y : List a) : a := (x.zip y).foldl (fun acc (a, b) => acc + a * b) 0

-- start positions should be non-negative; it is essentially an array index.
-- However, we need negative strides, so computing new starting positions will
-- involve arithmetic expressions involving adding negative numbers. For now
-- we assume we never underflow. In the future we will either log secretly
-- when this happens (e.g. via something like Util.dbgTrace) or put everything
-- in a logging monad.
def safeAdd (x : Nat) (y : Int) : Nat := (max 0 (x + y)).toNat
#guard safeAdd 7 (-1) == 6
#guard safeAdd 7 (-8) == 0

end Util

inductive ByteOrder where
| native
| littleEndian
| bigEndian
| notApplicable
deriving BEq, Repr, Inhabited


namespace ByteOrder

def isMultiByte (x : ByteOrder) : Bool := match x with
| littleEndian | bigEndian => true
| native | notApplicable => false

theorem littleEndianMultiByte : isMultiByte .littleEndian := by trivial

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

end ByteOrder

-- UInt8, UInt64, etc don't work well with bv_decide yet. It's coming
abbrev BV8 := BitVec 8

def UInt8.toBV8 (n : UInt8) : BV8 := BitVec.ofFin n.val

instance : Coe UInt8 BV8 where
  coe := UInt8.toBV8

def BV8.toUInt8 (n : BV8) : UInt8 := UInt8.ofNat n.toFin

instance : Coe BV8 UInt8 where
  coe := BV8.toUInt8

def BV8.toByteArray (x : BV8) : ByteArray := [x].toByteArray

def ByteArray.toBV8 (x : ByteArray) (startIndex : Nat) : Err BV8 :=
  let n := startIndex
  if H7 : n < x.size then
    let H0 : n + 0 < x.size := by linarith
    let x0 := x.get (Fin.mk _ H0)
    .ok (UInt8.toBV8 x0)
  else .error "Index out of range"

abbrev BV16 := BitVec 16

def BV16.toBytes (n : BV16) : BV8 × BV8 :=
  let n0 := (n >>> 0o00 &&& 0xFF).truncate 8
  let n1 := (n >>> 0o10 &&& 0xFF).truncate 8
  (n0, n1)

def BV16.ofBytes (x0 x1 : BV8) : BV16 :=
  (x0.zeroExtend 16 <<< 0o00) |||
  (x1.zeroExtend 16 <<< 0o10)

def ByteArray.toBV16 (x : ByteArray) (startIndex : Nat) (order : ByteOrder) : Err BV16 :=
  let n := startIndex
  if H7 : n + 1 < x.size then
    let H0 : n + 0 < x.size := by linarith
    let H1 : n + 1 < x.size := by linarith
    let x0 := x.get (Fin.mk _ H0)
    let x1 := x.get (Fin.mk _ H1)
    match order with
    | .notApplicable | .native => .error "illegal byte order"
    | .littleEndian => .ok (BV16.ofBytes x0 x1)
    | .bigEndian => .ok (BV16.ofBytes x1 x0)
  else .error "Index out of range"

def BV16.toByteArray (x : BV16) (ord : ByteOrder) (_H : ord.isMultiByte): ByteArray :=
  let (x0, x1) := x.toBytes
  match ord with
  | .littleEndian => [x0, x1].toByteArray
  | .bigEndian => [x1, x0].toByteArray

abbrev BV32 := BitVec 32

def BV32.toBytes (n : BV32) : BV8 × BV8 × BV8 × BV8 :=
  let n0 := (n >>> 0o00 &&& 0xFF).truncate 8
  let n1 := (n >>> 0o10 &&& 0xFF).truncate 8
  let n2 := (n >>> 0o20 &&& 0xFF).truncate 8
  let n3 := (n >>> 0o30 &&& 0xFF).truncate 8
  (n0, n1, n2, n3)

def BV32.ofBytes (x0 x1 x2 x3 : BV8) : BV32 :=
  (x0.zeroExtend 32 <<< 0o00) |||
  (x1.zeroExtend 32 <<< 0o10) |||
  (x2.zeroExtend 32 <<< 0o20) |||
  (x3.zeroExtend 32 <<< 0o30)

def ByteArray.toBV32 (x : ByteArray) (startIndex : Nat) (order : ByteOrder) : Err BV32 :=
  let n := startIndex
  if H7 : n + 3 < x.size then
    let H0 : n + 0 < x.size := by linarith
    let H1 : n + 1 < x.size := by linarith
    let H2 : n + 2 < x.size := by linarith
    let H3 : n + 3 < x.size := by linarith
    let x0 := x.get (Fin.mk _ H0)
    let x1 := x.get (Fin.mk _ H1)
    let x2 := x.get (Fin.mk _ H2)
    let x3 := x.get (Fin.mk _ H3)
    match order with
    | .notApplicable | .native => .error "illegal byte order"
    | .littleEndian => .ok (BV32.ofBytes x0 x1 x2 x3)
    | .bigEndian => .ok (BV32.ofBytes x3 x2 x1 x0)
  else .error "Index out of range"

def BV32.toByteArray (x : BV32) (ord : ByteOrder) (_H : ord.isMultiByte): ByteArray :=
  let (x0, x1, x2, x3) := x.toBytes
  match ord with
  | .littleEndian => [x0, x1, x2, x3].toByteArray
  | .bigEndian => [x3, x2, x1, x0].toByteArray

abbrev BV64 := BitVec 64

def BV64.toBytes (n : BV64) : BV8 × BV8 × BV8 × BV8 × BV8 × BV8 × BV8 × BV8 :=
  let n0 := (n >>> 0o00 &&& 0xFF).truncate 8
  let n1 := (n >>> 0o10 &&& 0xFF).truncate 8
  let n2 := (n >>> 0o20 &&& 0xFF).truncate 8
  let n3 := (n >>> 0o30 &&& 0xFF).truncate 8
  let n4 := (n >>> 0o40 &&& 0xFF).truncate 8
  let n5 := (n >>> 0o50 &&& 0xFF).truncate 8
  let n6 := (n >>> 0o60 &&& 0xFF).truncate 8
  let n7 := (n >>> 0o70 &&& 0xFF).truncate 8
  (n0, n1, n2, n3, n4, n5, n6, n7)

def BV64.ofBytes (x0 x1 x2 x3 x4 x5 x6 x7 : BV8) : BV64 :=
  (x0.zeroExtend 64 <<< 0o00) |||
  (x1.zeroExtend 64 <<< 0o10) |||
  (x2.zeroExtend 64 <<< 0o20) |||
  (x3.zeroExtend 64 <<< 0o30) |||
  (x4.zeroExtend 64 <<< 0o40) |||
  (x5.zeroExtend 64 <<< 0o50) |||
  (x6.zeroExtend 64 <<< 0o60) |||
  (x7.zeroExtend 64 <<< 0o70)

def ByteArray.toBV64 (x : ByteArray) (startIndex : Nat) (order : ByteOrder) : Err BV64 :=
  let n := startIndex
  if H7 : n + 7 < x.size then
    let H0 : n + 0 < x.size := by linarith
    let H1 : n + 1 < x.size := by linarith
    let H2 : n + 2 < x.size := by linarith
    let H3 : n + 3 < x.size := by linarith
    let H4 : n + 4 < x.size := by linarith
    let H5 : n + 5 < x.size := by linarith
    let H6 : n + 6 < x.size := by linarith
    let x0 := x.get (Fin.mk _ H0)
    let x1 := x.get (Fin.mk _ H1)
    let x2 := x.get (Fin.mk _ H2)
    let x3 := x.get (Fin.mk _ H3)
    let x4 := x.get (Fin.mk _ H4)
    let x5 := x.get (Fin.mk _ H5)
    let x6 := x.get (Fin.mk _ H6)
    let x7 := x.get (Fin.mk _ H7)
    match order with
    | .notApplicable | .native => .error "illegal byte order"
    | .littleEndian => .ok (BV64.ofBytes x0 x1 x2 x3 x4 x5 x6 x7)
    | .bigEndian => .ok (BV64.ofBytes x7 x6 x5 x4 x3 x2 x1 x0)
  else .error "Index out of range"

def BV64.toByteArray (x : BV64) (ord : ByteOrder) (_H : ord.isMultiByte): ByteArray :=
  let (x0, x1, x2, x3, x4, x5, x6, x7) := x.toBytes
  match ord with
  | .littleEndian => [x0, x1, x2, x3, x4, x5, x6, x7].toByteArray
  | .bigEndian => [x7, x6, x5, x4, x3, x2, x1, x0].toByteArray

theorem BV64.BytesRoundTrip (n : BV64) :
  let (x0, x1, x2, x3, x4, x5, x6, x7) := BV64.toBytes n
  let n' := BV64.ofBytes x0 x1 x2 x3 x4 x5 x6 x7
  n = n' := by
    unfold BV64.toBytes BV64.ofBytes
    bv_decide

theorem BV64.BytesRoundTrip1 (x0 x1 x2 x3 x4 x5 x6 x7 : BV8) :
  let n := BV64.ofBytes x0 x1 x2 x3 x4 x5 x6 x7
  let (x0', x1', x2', x3', x4', x5', x6', x7') := BV64.toBytes n
  x0 = x0' &&
  x1 = x1' &&
  x2 = x2' &&
  x3 = x3' &&
  x4 = x4' &&
  x5 = x5' &&
  x6 = x6' &&
  x7 = x7' := by
    unfold BV64.toBytes BV64.ofBytes
    bv_decide

#guard (
  let n : BV64 := 0x3FFAB851EB851EB8
  do
    let arr := n.toByteArray .littleEndian ByteOrder.littleEndianMultiByte
    let n' <- ByteArray.toBV64 arr 0 .littleEndian
    return n == n') == .ok true

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

end Dtype -- temporarily close the namespace so we don't duplicate the structure name

structure Dtype where
  name : Dtype.Name
  order : ByteOrder
deriving BEq, Repr, Inhabited

namespace Dtype

def byteOrderOk (x : Dtype) : Prop := !x.name.isMultiByte || (x.name.isMultiByte && x.order.isMultiByte)

def itemsize (t : Dtype) := t.name.itemsize

def fromString (s : String) : Err Dtype :=
  if s.length == 0 then .error "Empty dtype string" else
  do
    let order <- ByteOrder.fromChar (s.get 0)
    let name <- Name.fromString (s.drop 1)
    return { name, order }

def toString (t : Dtype) : String := t.order.toChar.toString.append t.name.toString

-- Examples
private def littleEndian (name : Name) : Dtype := { name, order := ByteOrder.littleEndian }
private def uint8 : Dtype := { name := Name.uint8 , order := ByteOrder.notApplicable }
private def uint16Little : Dtype := littleEndian Name.uint16
private def uint32Little : Dtype := littleEndian Name.uint32
private def uint64Little : Dtype := littleEndian Name.uint64

end Dtype

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

-- Shape is independent of the size of the dtype.
-/
abbrev Shape := List Nat

/-!
The strides are how many bytes you need to skip to get to the next element in that
"row". For example, in an array of 8-byte data with shape 2, 3, the strides are (24, 8).
Strides are also useful assuming 1-byte data for going back and forth between different
representations of the data. E.g. see `Tree` below.

Strides is typically used with the dtype size. However, it is also useful with byte size
1 to convert between different representations of the data.
-/
abbrev Strides := List Int -- Nonzero


-- The term "index" is overloaded a lot in NumPy. We don't have great alternative names,
-- so we're going with different extensions of the term. The `DimIndex` is the index in a
-- multi-dimensional array (as opposed to in the flat data, see `Position` below).
-- For example, in an array with shape (4, 2, 3), the `DimIndex` of the last element is (3, 1, 2)
-- DimIndex is independent of the size of the dtype.
abbrev DimIndex := List Nat

-- A `Position` is the index in the flat data array. For example, In an array with shape
-- 2, 3, the position of the element at (1, 2) is 5.
-- Position is independent of the size of the dtype.
abbrev Position := ℕ

inductive DataOrder where
| C
| Fortran
deriving Repr, Inhabited

namespace Shape

--! The number of elements in a tensor. All that's needed is the shape for this calculation.
-- TODO: cache this? Not sure how this works in Lean. Do we need to make Shape a struct and
-- put it in the record?

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
def unitStrides (s : Shape) (dataOrder : DataOrder) : Strides := match dataOrder with
| DataOrder.Fortran => [] -- Unimplemented
| DataOrder.C =>
  if H : s.isEmpty then [] else
  let s' := s.reverse
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
  let res : Strides := 1 :: loop s'.tail 1 (s'.head ok)
  res.reverse

def cUnitStrides (s : Shape) : Strides := s.unitStrides DataOrder.C

#guard cUnitStrides [2] == [1]
#guard cUnitStrides [2, 3] == [3, 1]
#guard cUnitStrides [2, 3, 5, 7] == [105, 35, 7, 1]

def sizedStrides (s : Shape) (dtype : Dtype) : Strides := List.map (fun x => x * dtype.itemsize) (cUnitStrides s)

-- Going from position to DimIndex is complicated by the possibility of
-- negative strides.
-- x x x x x x x x x x
--         ^         ^
--         p         s
-- For example, in the 1D array of length 10 above, the start position is at the end.
-- Assume, for example, that we obtained this array from the following sequence
-- # arange(10)[10 : 0 : -1]
-- #
def positionToDimIndex (strides : Strides) (n : Position) : DimIndex :=
  let foldFn acc stride :=
    let (posn, idx) := acc
    let div := (posn / stride).natAbs
    (Util.safeAdd posn (- div * stride), div :: idx)
  let (_, idx) := strides.foldl foldFn (n, [])
  idx.reverse

-- One-based strides
-- def dimIndexToPosition (strides : Strides) (index : DimIndex) : Position := Util.dot strides index

-- #guard positionToDimIndex [3, 1] 4 == [1, 1]
-- #guard dimIndexToPosition [3, 1] [1, 1] == 4

-- -- TODO: Make an iterator here rather than constructing the lists
-- def allDimIndices (shape : Shape) : List DimIndex :=
--   let strides := unitStrides shape DataOrder.C
--   let count := shape.count
--   let foldFn acc :=
--     let (position, indices) := acc
--     (position + 1, positionToDimIndex strides position :: indices)
--   let (position, indices) := shape.count.iterate foldFn (0, [])
--   if position < count then [] else indices.reverse

-- #guard allDimIndices [3, 2] == [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]

end Shape


/-!
This is the header of the on-disk Numpy format, typically with the .npy file extension.

https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#format-version-1-0
-/
structure NumpyHeader where
  -- Version is major.minor
  major : UInt8 := 1
  minor : UInt8 := 0
  -- Nat is definitely not the final representation. We need
  -- some function from python type descriptor and element to the appropriate
  -- Lean type.
  descr : Dtype
  -- Fortran order is column-major, C order is row major.
  -- Used for calculating the default strides
  -- TODO: make this an enum
  dataOrder : DataOrder := DataOrder.C
  shape : Shape
  deriving Repr, Inhabited

namespace NumpyHeader

/-!
A npy binary file has a header, some padding, then the data. This method computes the
size of the data portion of the file.
-/
def dataSize (header : NumpyHeader): Nat := header.descr.itemsize * header.shape.count

end NumpyHeader

/-!
A NumpyRepr is the data of the tensor, along with all metadata required to do
efficient computation. The `startIndex`, `endIndex`, and `strides` are inferred
from parsing or computed during a view creation.
-/
-- TODO: Add a `base` field to track aliasing? NumPy does this and it may make sense for us.
-- TODO: Do we want this to be inductive to handle array scalars? https://numpy.org/doc/stable/reference/arrays.scalars.html#arrays-scalars
--       Will forcing those into this type for now, but it seems wasteful.
-- TODO: Get useful info out of the header into the main struct
-- TODO: Get rid of endIndex. Not used. Access is controlled by startIndex and shape/stride/byte size only
structure NumpyRepr where
  header : NumpyHeader
  data : ByteArray
  startIndex : Nat := 0 -- TODO: Rename "index" here as "position". Index has too many other meanings. Pointer to the first byte of ndarray data, called `offset` in numpy
  endIndex : Nat := header.dataSize  -- Pointer to the byte after the last byte of ndarray data
  -- invariant: startIndex <= endIndex
  -- invariant: endIndex - startIndex = header.descr.numBytes * shape.count
  -- invariant: header.endIndex <= |data|
  -- invariant: header.dataSize <= |data|
  -- Strides independent of the data size
  unitStrides : Strides := header.shape.unitStrides header.dataOrder
  deriving Repr, Inhabited

namespace NumpyRepr

def empty (dtype : Dtype) (shape : Shape) : NumpyRepr :=
  let header : NumpyHeader := { descr := dtype, shape := shape }
  let data := ByteArray.mkEmpty (dtype.itemsize * shape.count)
  { header, data }

def dtype (x : NumpyRepr) : Dtype := x.header.descr

--! shape
def shape (x : NumpyRepr) : Shape := x.header.shape

--! number of dimensions
def ndim (x : NumpyRepr) : ℕ := x.shape.length

--! number of elements
def size (x : NumpyRepr) : ℕ := x.shape.count

--! number of bytes representing each element
def itemsize (x : NumpyRepr) : ℕ := x.header.descr.itemsize

def strides (x : NumpyRepr) : Strides := x.unitStrides.map (fun v => x.itemsize * v)

--! number of bytes representing the entire tensor
def nbytes (x : NumpyRepr) : ℕ := x.itemsize * x.size

-- Views can traverse backward through the data, so we need to check both the front
-- and the back of the data to ensure a position is in range
def inRange (x : NumpyRepr) (n : ℕ) : Bool := x.startIndex <= n && n < x.endIndex

--! Reshaping a tensor is just re-interpreting the elements in a different order.
-- FIXME: The unit strides of the input should be inputs into the stride calculation
-- for the reshaped array. I'm not quite sure how this works yet.
def reshape (x : NumpyRepr) (shape : Shape) : Err NumpyRepr := do
  if x.shape.count == shape.count then
    return { x with header.shape := shape, unitStrides := shape.unitStrides x.header.dataOrder }
  else
    .error "Reshaping must have the same number of implied elements"

def reshape! (x : NumpyRepr) (shape : Shape) : NumpyRepr :=
  match reshape x shape with
  | .error msg => panic msg
  | .ok x => x

class TensorElement (a : Type) where
  dtype : Dtype
  itemsize : Nat
  ofNat : Nat -> a
  toByteArray (x : a) : ByteArray
  fromByteArray (arr : ByteArray) (startIndex : Nat) : Err a

namespace TensorElement

-- An array-scalar is a box around a scalar with nil shape that can be used for array operations like broadcasting
def arrayScalar [w : TensorElement a] (x : a) : NumpyRepr :=
  let header := { descr := w.dtype, shape := [] }
  let data := w.toByteArray x
  { header, data }

--! An array of the numbers from 0 to n-1
--! https://numpy.org/doc/2.1/reference/generated/numpy.arange.html
def arange (a : Type) [w : TensorElement a] (n : Nat) : NumpyRepr :=
  let header := { descr := w.dtype, shape := [n] }
  let data := ByteArray.mkEmpty (n * w.itemsize)
  let foldFn i data :=
    let bytes := w.toByteArray (w.ofNat i)
    ByteArray.copySlice bytes 0 data (i * w.itemsize) w.itemsize
  let data := Nat.fold foldFn n data
  { header, data }

instance BV8Native : TensorElement BV8 where
  dtype := Dtype.mk .uint8 .native
  itemsize := 1
  ofNat n := n
  toByteArray (x : BV8) : ByteArray := x.toByteArray
  fromByteArray arr startIndex := ByteArray.toBV8 arr startIndex

instance BV16Little : TensorElement BV16 where
  dtype := Dtype.mk .uint16 .littleEndian
  itemsize := 2
  ofNat n := n
  toByteArray (x : BV16) : ByteArray := x.toByteArray .littleEndian ByteOrder.littleEndianMultiByte
  fromByteArray arr startIndex := ByteArray.toBV16 arr startIndex .littleEndian

instance BV32Little : TensorElement BV32 where
  dtype := Dtype.mk .uint32 .littleEndian
  itemsize := 4
  ofNat n := n
  toByteArray (x : BV32) : ByteArray := x.toByteArray .littleEndian ByteOrder.littleEndianMultiByte
  fromByteArray arr startIndex := ByteArray.toBV32 arr startIndex .littleEndian

instance BV64Little : TensorElement BV64 where
  dtype := Dtype.mk .uint64 .littleEndian
  itemsize := 8
  ofNat n := n
  toByteArray (x : BV64) : ByteArray := x.toByteArray .littleEndian ByteOrder.littleEndianMultiByte
  fromByteArray arr startIndex := ByteArray.toBV64 arr startIndex .littleEndian

#eval arange BV16 10

end TensorElement

def transpose (x : NumpyRepr) : NumpyRepr :=
  let shape := x.shape.reverse
  let header := { x.header with shape }
  let unitStrides := x.unitStrides.reverse
  { x with header, unitStrides }

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
