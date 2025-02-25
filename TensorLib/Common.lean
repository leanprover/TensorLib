/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/

import Std.Tactic.BVDecide
import Plausible
open Plausible

namespace TensorLib

private def plausibleDefaultConfig : Plausible.Configuration := {}
/-
This suppresses messages from Plausible on the commandline when building, such as
    info: ././././TensorLib/Common.lean:37:0: Unable to find a counter-example
If you want to see the counterexample in the IDE, you need to remove the configuration argument,
but remember to add it back once you've fixed the bug.
-/
def cfg : Plausible.Configuration := { plausibleDefaultConfig with quiet := true }

--! The error monad for TensorLib
abbrev Err := Except String

def impossible {a : Type} [h : Inhabited a] (msg : String := "") := @panic a h s!"Invariant violation: {msg}"

instance [BEq a] : BEq (Err a) where
  beq x y := match x, y with
  | .ok x, .ok y => x == y
  | .error x, .error y => x == y
  | _, _ => false

def get! [Inhabited a] (x : Err a) : a := match x with
| .error msg => impossible msg
| .ok x => x

def dot [Add a][Mul a][Zero a] (x y : List a) : a := (x.zip y).foldl (fun acc (a, b) => acc + a * b) 0

def natDivCeil (num denom : Nat) : Nat := (num + denom - 1) / denom

section Test

#eval Testable.check (cfg := cfg) (
  ∀ (x y : Nat),
  let c := natDivCeil x y
  let f := x / y
  c == f || c == (f + 1)
)

local instance : SampleableExt (Nat × Nat) :=
  SampleableExt.mkSelfContained do
    let x ← SampleableExt.interpSample Nat
    let n <- SampleableExt.interpSample Nat
    return (x * n, x)

#eval Testable.check (cfg := cfg) (
  ∀ (xy : Nat × Nat),
  let (x, y) := xy
  x % y = 0 → x / y = natDivCeil x y
)

end Test

def natProd (shape : List Nat) : Nat := shape.foldl (fun x y => x * y) 1


-- We generally have large tensors, so don't show them by default
instance ByteArrayRepr : Repr ByteArray where
  reprPrec x _ :=
    if x.size < 100 then x.toList.repr 100 else
    s!"ByteArray of size {x.size}"

/-!
NumPy arrays can be stored in big-endian or little-endian order on disk, regardless
of the architecture of the machine saving the array. Since we read these arrays
into memory at certain data types, for multi-byte data types we need to know the
endian-ness.
-/
inductive ByteOrder where
| oneByte
| littleEndian
| bigEndian
deriving BEq, Repr, Inhabited

namespace ByteOrder

@[simp]
def isMultiByte (x : ByteOrder) : Bool := match x with
| .oneByte => false
| .littleEndian | .bigEndian => true

def bytesToNat (order : ByteOrder) (bytes : ByteArray) : Nat := Id.run do
  let mut n : Nat := 0
  let nbytes := bytes.size
  for i in [0:nbytes] do
    let v : UInt8 := bytes.get! i
    let p := match order with
    | .oneByte => 0 -- nbytes = 1
    | .littleEndian => i
    | .bigEndian => nbytes - 1 - i
    n := n + Pow.pow 2 (8 * p) * v.toNat
  return n

#guard bytesToNat .littleEndian (ByteArray.mk #[1, 1]) == 257
#guard bytesToNat .bigEndian (ByteArray.mk #[1, 1]) == 257
#guard bytesToNat .littleEndian (ByteArray.mk #[0, 1]) == 256
#guard bytesToNat .bigEndian (ByteArray.mk #[0, 1]) == 1
#guard bytesToNat .littleEndian (ByteArray.mk #[0xFF, 0xFF]) == 65535
#guard bytesToNat .bigEndian (ByteArray.mk #[0xFF, 0xFF]) == 65535
#guard bytesToNat .bigEndian (ByteArray.mk #[0x80, 0]) == 32768
#guard bytesToNat .littleEndian (ByteArray.mk #[0x80, 0]) == 0x80

def bytesToInt (order : ByteOrder) (bytes : ByteArray) : Int := Id.run do
  let mut n : Nat := 0
  let nbytes := bytes.size
  let signByte := match order with
  | .littleEndian => bytes.get! (nbytes - 1)
  | .bigEndian | oneByte => bytes.get! 0
  let negative := 128 <= signByte
  for i in [0:nbytes] do
    let v : UInt8 := bytes.get! i
    let v := if negative then UInt8.complement v else v
    let p := match order with
    | .oneByte => 0 -- nbytes = 1
    | .littleEndian => i
    | .bigEndian => nbytes - 1 - i
    n := n + Pow.pow 2 (8 * p) * v.toNat
  return if 128 <= signByte then -(n + 1) else n

#guard bytesToInt .littleEndian (ByteArray.mk #[1, 1]) == 257
#guard bytesToInt .bigEndian (ByteArray.mk #[1, 1]) == 257
#guard bytesToInt .littleEndian (ByteArray.mk #[0, 1]) == 256
#guard bytesToInt .bigEndian (ByteArray.mk #[0, 1]) == 1
#guard bytesToInt .littleEndian (ByteArray.mk #[0xFF, 0xFF]) == -1
#guard bytesToInt .bigEndian (ByteArray.mk #[0xFF, 0xFF]) == -1
#guard bytesToInt .bigEndian (ByteArray.mk #[0x80, 0]) == -32768
#guard bytesToInt .littleEndian (ByteArray.mk #[0x80, 0]) == 0x80

def bitVecToByteArray (order : ByteOrder) (n : Nat) (v : BitVec n) : ByteArray := Id.run do
  let numBytes := natDivCeil n 8
  let mut arr := ByteArray.mkEmpty numBytes
  match order with
  | .oneByte =>
    let byte := (v &&& 0xFF).toNat.toUInt8
    return arr.push byte
  | .littleEndian =>
    for i in [0 : numBytes] do
      let byte := (v.ushiftRight (i * 8) &&& 0xFF).toNat.toUInt8
      arr := arr.push byte
    return arr
  | .bigEndian =>
    for i in [0 : numBytes] do
      let byte := (v.ushiftRight ((numBytes - i - 1) * 8) &&& 0xFF).toNat.toUInt8
      arr := arr.push byte
    return arr

#guard (bitVecToByteArray .bigEndian 16 0x0100).toList == [1, 0]
#guard (bitVecToByteArray .littleEndian 16 0x0100).toList == [0, 1]
#guard (bitVecToByteArray .bigEndian 20 0x01000).toList == [0, 16, 0]
#guard (bitVecToByteArray .littleEndian 32 0x1).toList == [1, 0, 0, 0]
#guard (bitVecToByteArray .bigEndian 32 0x1).toList == [0, 0, 0, 1]

end ByteOrder

/-!
The strides are how many bytes you need to skip to get to the next element in that
"row". For example, in an array of 8-byte data with shape 2, 3, the strides are (24, 8).
Strides are also useful assuming 1-byte data for going back and forth between different
representations of the data. E.g. see `Tree` below.

Strides is typically used with the dtype size. However, it is also useful with byte size
1 to convert between different representations of the data. We call strides of the later
form "unit strides". So for example, a 1D array of f32 with normal indexing would have
a stride of 4 and a unit stride of 1. An f16 would have stride 2, unit stride 1.
-/
abbrev Strides := List Int

-- The term "index" is overloaded a lot in NumPy. We don't have great alternative names,
-- so we're going with different extensions of the term. The `DimIndex` is the index in a
-- multi-dimensional array (as opposed to in the flat data, see `Position` below).
-- For example, in an array with shape (4, 2, 3), the `DimIndex` of the last element is (3, 1, 2)
-- DimIndex is independent of the size of the dtype.
abbrev DimIndex := List Nat

/-!
A `Position` is the index in the flat data array. For example, In an array with shape
2, 3, the position of the element at (1, 2) is 5.
Position is independent of the size of the dtype.
-/
abbrev Position := Nat

/-!
A `Offset` is the positional offset from the starting position. For negative values, the magnitude
should be smaller than the starting position. So for tensor `t`, `0 <= t.startingPosition + offset < t.count`.
-/
abbrev Offset := Int

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

Shape is independent of the size of the dtype.
TODO: Consider adding Coe instance for List Nat?
-/
structure Shape where
  val : List Nat
deriving BEq, Repr, Inhabited

namespace Shape

instance : ToString Shape where
  toString := reprStr

def empty : Shape := Shape.mk []

def append (shape : Shape) (dims : List Nat) : Shape := Shape.mk (shape.val ++ dims)

--! The number of elements in a tensor. All that's needed is the shape for this calculation.
-- TODO: Put this in the struct?
def count (shape : Shape) : Nat := natProd shape.val

--! Number of dimensions
def ndim (shape : Shape) : Nat := shape.val.length

def map (shape : Shape) (f : List Nat -> List Nat) : Shape := Shape.mk (f shape.val)

def dimIndexInRange (shape : Shape) (dimIndex : DimIndex) :=
  shape.ndim == dimIndex.length &&
  (shape.val.zip dimIndex).all fun (n, i) => i < n

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
def unitStrides (s : Shape) : Strides :=
  let s := s.val
  if H : s.isEmpty then [] else
  let s' := s.reverse
  let rec loop (xs : List Nat) (lastShape lastDimSize : Nat): List Int := match xs with
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

#guard unitStrides (Shape.mk [2]) == [1]
#guard unitStrides (Shape.mk [2, 3]) == [3, 1]
#guard unitStrides (Shape.mk [2, 3, 5, 7]) == [105, 35, 7, 1]

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
    ((posn + (- div * stride)).toNat, div :: idx)
  let (_, idx) := strides.foldl foldFn (n, [])
  idx.reverse

-- TODO: Return `Err Offset` for when the strides and index have different lengths?
def dimIndexToOffset (strides : Strides) (index : DimIndex) : Offset := dot strides (index.map Int.ofNat)

#guard positionToDimIndex [3, 1] 4 == [1, 1]
#guard dimIndexToOffset [3, 1] [1, 1] == 4

-- In general you should use DimsIter instead of this, which is equivalent
-- to `DimsIter.toList` but I left it here because it is obviously terminating.
def allDimIndices (shape : Shape) : List DimIndex := Id.run do
  let strides := unitStrides shape
  let count := shape.count
  let mut indices := []
  for i in [0:count] do
    indices := positionToDimIndex strides i :: indices
  return indices.reverse

#guard allDimIndices (Shape.mk [5]) == [[0], [1], [2], [3], [4]]
#guard allDimIndices (Shape.mk [3, 2]) == [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]

-- NumPy supports negative indices, which simply wrap around. E.g. `x[.., -1, ..] = x[.., n-1, ..]` where `n` is the
-- dimension in question. It only supports `-n` to `n`.
def intIndexToDimIndex (shape : Shape) (index : List Int) : Err DimIndex := do
  if shape.ndim != index.length then .error "intsToDimIndex length mismatch" else
  let conv (dim : Nat) (ind : Int) : Err Nat :=
    if 0 <= ind then
      if ind < dim then .ok ind.toNat
      else .error "index out of bounds"
    else if ind < -dim then .error "index out of bounds"
    else .ok (dim + ind).toNat
  (shape.val.zip index).mapM (fun (dim, ind) => conv dim ind)

#guard intIndexToDimIndex (Shape.mk [1, 2, 3]) [0, -1, -1] == (.ok [0, 1, 2])
#guard intIndexToDimIndex (Shape.mk [1, 2, 3]) [0, 1, -2] == (.ok [0, 1, 1])

end Shape

/-
We store the upper limits backwards so we can have access to the one moving
fastest at the left of the list. Because this could cause confusion, we make
the constructor and fields private.

`curr` has not yet been returned by `next`
-/
structure DimsIter where
  private mk ::
  private dims : List Nat
  private curr : List Nat
deriving Inhabited
-- max.length = cur.length

namespace DimsIter

--! Total number of elements in the iterator, assuming we begin at the
-- 0th value (all 0s)
def size (iter : DimsIter) : Nat := natProd iter.dims

def make (dims : Shape) : DimsIter :=
  DimsIter.mk dims.val.reverse (List.replicate dims.ndim 0)

/-
We include `hasNext` for testing and documentation. We don't use it
in functions because we have the `size` field to tell us when we're
done, and that admits structural termination, unlike this method.

This is trickier than I would like. The final value is
[dim0-1, dim1-1, dim2-1, ..., dimN-1], but it hasn't been
returned yet. To signal we are done, we'll set the final value
to dimN.
-/
private def hasNext (iter : DimsIter) : Bool :=
  let rec loop (dims ns : List Nat) : Bool :=
    match dims, ns with
    | dim :: dims, n :: ns => n < dim && loop dims ns
    | _, _ => true
  loop iter.dims iter.curr

#guard (DimsIter.mk [1] [0]).hasNext -- [0] hasn't been returned yet
#guard !(DimsIter.mk [1] [1]).hasNext
#guard (DimsIter.mk [1, 3] [0, 2]).hasNext
#guard !(DimsIter.mk [1, 3] [0, 3]).hasNext

/-
Return the next element. Note that we don't need a `hasNext` predicate because
we have a `size` field which tells us when to stop in an iteration loop.

Note: I tried writing this as a `do/for` loop and in this case the recursive
one seems nicer. We are walking over two lists simultaneously, which is easy
here but with a for loop is either quadratic or awkward.
-/
def next (iter : DimsIter) : DimIndex × DimsIter :=
  -- Invariant: `acc` is a list of 0s, so doesn't need to be reversed
  let rec loop (acc ms ns : List Nat) : List Nat :=
    match ms, ns with
      -- this case is to bump the final value to _dim, rather than _dim - 1 which
      -- is the highest the other indexes get
    | [_dim], [n] =>
      ((n + 1) :: acc).reverse
    | dim :: dims, n :: ns =>
      if n < dim - 1 then ((n + 1) :: acc).reverse.append ns
      else loop (0 :: acc) dims ns
    | _, _ => []
  let curr' := loop [] iter.dims iter.curr
  (iter.curr.reverse, { iter with curr := curr' })

instance [Monad m] : ForIn m DimsIter DimIndex where
  forIn {α} [Monad m] (iter : DimsIter) (x : α) (f : List Nat -> α -> m (ForInStep α)) : m α := do
    let mut iter := iter
    let mut res := x
    for _ in [0:iter.size] do
      let (dims, iter') := iter.next
      iter := iter'
      match <- f dims res with
      | .yield k => res := k
      | .done k => return k
    return res

private def toList (iter : DimsIter) : List DimIndex := Id.run do
  let mut res := []
  for xs in iter do
    res := xs :: res
  return res.reverse

#guard (DimsIter.make $ Shape.mk [0, 1] ).toList == []
#guard (DimsIter.make $ Shape.mk [1, 0]).toList == []
#guard (DimsIter.make $ Shape.mk [1]).toList == [[0]]
#guard (DimsIter.make $ Shape.mk [3]).toList == [[0], [1], [2]]
#guard (DimsIter.make $ Shape.mk [1, 1]).toList == [[0, 0]]
#guard (DimsIter.make $ Shape.mk [2, 1]).toList == [[0, 0], [1, 0]]
#guard (DimsIter.make $ Shape.mk [1, 1, 1]).toList == [[0, 0, 0]]
#guard (DimsIter.make $ Shape.mk [1, 1, 2]).toList == [[0, 0, 0], [0, 0, 1]]
#guard (DimsIter.make $ Shape.mk [3, 2]).toList == [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]

private def testBreak (iter : DimsIter) : List DimIndex := Id.run do
  let mut res := []
  for xs in iter do
    res := xs :: res
    break
  return res.reverse

#guard (DimsIter.make $ Shape.mk [3, 2]).testBreak == [[0, 0]]

private def testReturn (iter : DimsIter) : List DimIndex := Id.run do
  let mut res := []
  let mut i := 0
  for xs in iter do
    res := xs :: res
    i := i + 1
    if i == 3 then return res.reverse
  return res.reverse

#guard (DimsIter.make $ Shape.mk [3, 2]).testReturn == [[0, 0], [0, 1], [1, 0]]

end DimsIter

abbrev BV8 := BitVec 8

def BV8.ofNat (i : Nat) : BV8 := i.toUInt8.toBitVec

-- `_root_` is required to add dot methods to UInt8, which is outside TensorLib
def _root_.UInt8.toBV8 (n : UInt8) : BV8 := BitVec.ofFin n.val
def BV8.toUInt8 (n : BV8) : UInt8 := UInt8.ofNat n.toFin

def BV8.toByteArray (x : BV8) : ByteArray := [x.toUInt8].toByteArray

def ByteArray.toBV8 (x : ByteArray) (startIndex : Nat) : Err BV8 :=
  let n := startIndex
  if H7 : n < x.size then
    let H0 : n + 0 < x.size := by omega
    let x0 := x.get (Fin.mk _ H0)
    .ok (UInt8.toBV8 x0)
  else .error s!"Index out of range: {n}"

abbrev BV16 := BitVec 16

def BV16.toBytes (n : BV16) : BV8 × BV8 :=
  let n0 := (n >>> 0o00 &&& 0xFF).truncate 8
  let n1 := (n >>> 0o10 &&& 0xFF).truncate 8
  (n0, n1)

def BV16.ofBytes (x0 x1 : BV8) : BV16 :=
  (x0.zeroExtend 16 <<< 0o00) |||
  (x1.zeroExtend 16 <<< 0o10)

theorem BV16.BytesRoundTrip (n : BV16) :
  let (x0, x1) := BV16.toBytes n
  let n' := BV16.ofBytes x0 x1
  n = n' := by
    unfold BV16.toBytes BV16.ofBytes
    bv_decide

theorem BV16.BytesRoundTrip1 (x0 x1 : BV8) :
  let n := BV16.ofBytes x0 x1
  let (x0', x1') := BV16.toBytes n
  x0 = x0' &&
  x1 = x1' := by
    unfold BV16.toBytes BV16.ofBytes
    bv_decide

def ByteArray.toBV16 (x : ByteArray) (startIndex : Nat) (order : ByteOrder) : Err BV16 :=
  let n := startIndex
  if H7 : n + 1 < x.size then
    let H0 : n + 0 < x.size := by omega
    let H1 : n + 1 < x.size := by omega
    let x0 := x.get (Fin.mk _ H0)
    let x1 := x.get (Fin.mk _ H1)
    match order with
    | .oneByte => .error "illegal byte order"
    | .littleEndian => .ok (BV16.ofBytes x0.toBV8 x1.toBV8)
    | .bigEndian => .ok (BV16.ofBytes x1.toBV8 x0.toBV8)
  else .error s!"Index out of range: {n}"

def BV16.toByteArray (x : BV16) (ord : ByteOrder) : ByteArray :=
  let (x0, x1) := x.toBytes
  let arr := match ord with
  | .littleEndian => [x0, x1]
  | .bigEndian => [x1, x0]
  | .oneByte => [] -- Avoid Err for now
  (arr.map BV8.toUInt8).toByteArray

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

theorem BV32.BytesRoundTrip (n : BV32) :
  let (x0, x1, x2, x3) := BV32.toBytes n
  let n' := BV32.ofBytes x0 x1 x2 x3
  n = n' := by
    unfold BV32.toBytes BV32.ofBytes
    bv_decide

theorem BV32.BytesRoundTrip1 (x0 x1 x2 x3 : BV8) :
  let n := BV32.ofBytes x0 x1 x2 x3
  let (x0', x1', x2', x3') := BV32.toBytes n
  x0 = x0' &&
  x1 = x1' &&
  x2 = x2' &&
  x3 = x3' := by
    unfold BV32.toBytes BV32.ofBytes
    bv_decide

def ByteArray.toBV32 (x : ByteArray) (startIndex : Nat) (order : ByteOrder) : Err BV32 :=
  let n := startIndex
  if H7 : n + 3 < x.size then
    let H0 : n + 0 < x.size := by omega
    let H1 : n + 1 < x.size := by omega
    let H2 : n + 2 < x.size := by omega
    let H3 : n + 3 < x.size := by omega
    let x0 := x.get (Fin.mk _ H0)
    let x1 := x.get (Fin.mk _ H1)
    let x2 := x.get (Fin.mk _ H2)
    let x3 := x.get (Fin.mk _ H3)
    match order with
    | .oneByte => .error "illegal byte order"
    | .littleEndian => .ok (BV32.ofBytes x0.toBV8 x1.toBV8 x2.toBV8 x3.toBV8)
    | .bigEndian => .ok (BV32.ofBytes x3.toBV8 x2.toBV8 x1.toBV8 x0.toBV8)
else .error s!"Index out of range: {n}"

def BV32.toByteArray (x : BV32) (ord : ByteOrder) : ByteArray :=
  let (x0, x1, x2, x3) := x.toBytes
  let arr := match ord with
  | .littleEndian => [x0, x1, x2, x3]
  | .bigEndian => [x3, x2, x1, x0]
  | .oneByte => []
  (arr.map BV8.toUInt8).toByteArray

abbrev BV64 := BitVec 64

def BV64.ofNat (i : Nat) : BV64 := i.toUInt64.toBitVec

def BV64.ofInt (i : Int) : BV64 := i.toInt64.toBitVec

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

def ByteArray.toBV64 (x : ByteArray) (startIndex : Nat) (order : ByteOrder) : Err BV64 :=
  let n := startIndex
  if H7 : n + 7 < x.size then
    let H0 : n + 0 < x.size := by omega
    let H1 : n + 1 < x.size := by omega
    let H2 : n + 2 < x.size := by omega
    let H3 : n + 3 < x.size := by omega
    let H4 : n + 4 < x.size := by omega
    let H5 : n + 5 < x.size := by omega
    let H6 : n + 6 < x.size := by omega
    let x0 := x.get (Fin.mk _ H0)
    let x1 := x.get (Fin.mk _ H1)
    let x2 := x.get (Fin.mk _ H2)
    let x3 := x.get (Fin.mk _ H3)
    let x4 := x.get (Fin.mk _ H4)
    let x5 := x.get (Fin.mk _ H5)
    let x6 := x.get (Fin.mk _ H6)
    let x7 := x.get (Fin.mk _ H7)
    match order with
    | .oneByte => .error "illegal byte order"
    | .littleEndian => .ok (BV64.ofBytes x0.toBV8 x1.toBV8 x2.toBV8 x3.toBV8 x4.toBV8 x5.toBV8 x6.toBV8 x7.toBV8)
    | .bigEndian => .ok (BV64.ofBytes x7.toBV8 x6.toBV8 x5.toBV8 x4.toBV8 x3.toBV8 x2.toBV8 x1.toBV8 x0.toBV8)
  else .error s!"Index out of range: {n}"

def BV64.toByteArray (x : BV64) (ord : ByteOrder) : ByteArray :=
  let (x0, x1, x2, x3, x4, x5, x6, x7) := x.toBytes
  let arr := match ord with
  | .littleEndian => [x0, x1, x2, x3, x4, x5, x6, x7]
  | .bigEndian => [x7, x6, x5, x4, x3, x2, x1, x0]
  | .oneByte => []
  (arr.map BV8.toUInt8).toByteArray

#guard (
  let n : BV64 := 0x3FFAB851EB851EB8
  do
    let arr := n.toByteArray .littleEndian
    let n' <- ByteArray.toBV64 arr 0 .littleEndian
    return n == n') == .ok true

end TensorLib
