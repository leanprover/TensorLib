import Init.System.IO
import Init.System.FilePath
import Mathlib.Tactic
import TensorLib.NonEmptyList
import TensorLib.TensorElement

/-!
We largely duplicate the NumPy representation of tensors.

The binary format is described here: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
and here: https://github.com/numpy/numpy/blob/067cb067cb17a20422e51da908920a4fbb3ab851/doc/neps/nep-0001-npy-format.rst

In addition to being efficient, this allows us to directly parse .npy input files.
-/

namespace TensorLib

/-! The subset of types NumPy supports that we care about -/
inductive NumpyDtype where
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
  deriving Repr

namespace NumpyDtype

--! Number of bytes used by each element of the given dtype
def bytes (x: NumpyDtype): Nat := match x with
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
def fromString (s: String) : Except String NumpyDtype := match s with
| "<b1" | "|b1"=> .ok bool
| "|i1" => .ok int8
| "<i2" => .ok int16
| "<i4" => .ok int32
| "<i8" => .ok int64
| "|u1" => .ok uint8
| "<u2" => .ok uint16
| "<u4" => .ok uint32
| "<u8" => .ok uint64
| "<f2" => .ok float16
| "<f4" => .ok float32
| "<f8" => .ok float64
| _ => .error s!"Can't parse {s} as a dtype"

end NumpyDtype

--! Numpy disk format uses python convention for bools; True/False
def NumpyBool.fromString (s: String): Except String Bool := match s with
| "True" => .ok true
| "False" => .ok false
| _ => .error s!"Can't parse {s} as Bool"

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

The only way to have an empty shape in Numpy is as

```
>>> np.array(None).shape
()
```

We can't write None in Lean, nor do we want to. If we allow empty lists in shape computation, we
will need to continually handle the [] case with something like

```
match shape with
| [] => impossible
| x :: xs => E(x, xs)
```

As a result, we would either be in an error monad constantly for  no reason, or need to pass
non-emptiness proofs around. For now, we avoid this case using NonemptyList.
-/
abbrev Shape := NonEmptyList Nat
abbrev Strides := NonEmptyList Nat

namespace Shape

--! The number of elements in a tensor. All that's needed is the shape for this calculation.
def count (s : Shape) : Nat := match s with
| .mk x xs => xs.foldl (fun x y => x * y) x

/-!
Strides can be computed from the shape by figuring out how many elements you
need to jump over to get to the next spot and mulitplying by the bytes in each
element.

A given shape can have different strides if the tensor is a view of another
tensor. For example, in a square matrix, the transposed matrix view has the same
shape but the strides change.
-/
def defaultStrides (dtype : NumpyDtype) (s : Shape) : Strides :=
  let s := s.reverse
  let bytes := dtype.bytes
  let rec loop (xs : List ℕ) (lastShape lastDimSize : ℕ): List ℕ := match xs with
  | [] => []
  | d :: ds =>
    let rest := loop ds (lastShape * lastDimSize) d
    lastShape * lastDimSize :: rest
  let res : Strides := { hd := dtype.bytes, tl := loop s.tl bytes s.hd }
  res.reverse

#eval defaultStrides NumpyDtype.uint32 [2]
#eval defaultStrides NumpyDtype.uint32 [2, 3]
#eval defaultStrides NumpyDtype.uint32 [2, 3, 5, 7]

end Shape

/-!
This is the header of the on-disk Numpy format, typically with the .npy file extension.

https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#format-version-1-0
-/
structure NumpyHeader where
  -- Version is major.minor
  major : Nat
  minor : Nat
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
def dataSize (header : NumpyHeader): Nat := header.descr.bytes * header.shape.count

end NumpyHeader

-- We generally have huge tensors, so don't show them by default
instance ByteArrayRepr : Repr ByteArray where
  reprPrec x _ :=
    let s := toString x.size
    s!"ByteArray of size {s}"

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
  -- invariant: endIndex - startIndex = descr.bytes * shape.count
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
def itemsize (x : NumpyRepr) : ℕ := x.header.descr.bytes

--! number of bytes representing the entire tensor
def nbytes (x : NumpyRepr) : ℕ := x.itemsize * x.size

section Shape

abbrev M a := Except String a

/-!
Reshaping a tensor is just re-interpreting the elements in
a different order. This will also impact the iteration order/strides
when we implement those.
-/
def reshape (x : NumpyRepr) (shape : Shape) : M NumpyRepr := do
  if x.shape.count == shape.count then
    return { x with header.shape := shape }
  else
    .error "Reshaping must have the same number of implied elements"

end Shape

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
  then { b with left := .appendListL (List.replicate (n2 - n1) 1) b.left }
  else { b with right := .appendListL (List.replicate (n1 - n2) 1) b.right }

private theorem oneExtendPrefixLength (b : Broadcast) :
  let b' := oneExtendPrefix b
  b'.left.length = b'.right.length := by
  cases b
  rename_i left right
  simp [oneExtendPrefix]
  by_cases H : left.length <= right.length
  . simp [H]
    rw [NonEmptyList.appendListLLength, List.length_replicate, Nat.sub_add_cancel]
    exact H
  . simp [H]
    rw [NonEmptyList.hAppendListLLength, List.length_replicate, Nat.sub_add_cancel]
    linarith

private def matchPairs (b : Broadcast) : Option Shape :=
  if b.left.length != b.right.length then none else
  let f xy := match xy with
    | (x, y) =>
      if x == y then some x
      else if x == 1 then some y
      else if y == 1 then some x
      else none
  traverse f (NonEmptyList.zip b.left b.right)

--! Returns the shape resulting from broadcast the arguments
def broadcast (b : Broadcast) : Option Shape := matchPairs (oneExtendPrefix b)

--! Whether broadcasting is possible
def canBroadcast (b : Broadcast) : Bool := (broadcast b).isSome

#eval matchPairs (Broadcast.mk [1, 2, 3] [7, 2, 1])
#eval broadcast (Broadcast.mk [1, 2, 3] [7, 7, 9, 2, 1])

-- todo: add plausible properties when property-based testing settles down in Lean-land
#guard
 let x1 := [1,2,3]
 let x2 := [2,3]
 let b1 := Broadcast.mk x1 x1
 let b2 := Broadcast.mk x1 x2
 oneExtendPrefix b1 == b1 &&
 broadcast b2 == broadcast b1 &&
 broadcast b2 == some (.fromList! [1, 2, 3])

end Broadcast


--! Parse a .npy file
namespace Parse

structure ParsingState where
  source : ByteArray    -- Source data being parsed
  index : Nat           -- Index into source data
  headerEnd : Nat
  descr : Option NumpyDtype
  fortranOrder : Option Bool
  shape : Option Shape
  debug : List String
  deriving Repr

abbrev PState (T : Type) := EStateM String ParsingState T

instance : MonadLiftT (Except String) PState where
  monadLift x := match x with
  | .ok x => .ok x
  | .error x => .error x

-- Except is nicer sometimes
def resultExcept {σ : Type} (x : EStateM.Result String σ a) : Except String a := match x with
| .ok x _ => .ok x
| .error x _ => .error x

variable {T : Type}

def debug (msg : String) : PState Unit := do
  modify (fun s => { s with debug := msg :: s.debug })

-- Whitespace is under-specified in the docs linked above, but I think
-- Lean's isWhitespace will be fine here. I have never seen a non 0x20
-- space character in my experiments.
def whitespace : PState Unit := do
  let s <- get
  let mut count := 0
  for i in [s.index : s.headerEnd] do
    let c := Char.ofUInt8 s.source[i]!
    if c.isWhitespace then
      count := count + 1
    else
      break
  set { s with index := s.index + count }

def tryParse (p : PState T) : PState (Option T) := fun s =>
  match p s with
  | .error _ s => .ok none s
  | .ok x s => .ok (some x) s

def ignore (p : PState T) : PState Unit := do
  let _ <- tryParse p
  return ()

-- Does some heavy lifting, parsing both normal ids (map keys) and weird
-- values like the dtypes
def parseToken : PState String := do
  whitespace
  let s ← get
  let mut token := ""
  for i in [s.index : s.headerEnd] do
    let b := s.source.get! i
    let c := Char.ofUInt8 b
    if c.isAlphanum || c = '_' || c = '<' || c = '|' then token := token.push c
    else break
  if token.length != 0 then
    set { s with index := s.index + token.length }
    return token
  else
    .error "Can't parse token"

def consume (c: Char) : PState Unit := do
  whitespace
  let s <- get
  let i := s.index
  if s.source[i]! == c.toUInt8 then
    set { s with index := i + 1 }
  else
    .error s!"Can't consume {c}. So far we have {repr s}"

def quote : PState Unit := consume '\''
def colon : PState Unit := consume ':'
def comma : PState Unit := consume ','

def quoted (p : PState T) : PState T := do
  quote
  let x <- p
  quote
  return x

partial def parseCommaListAux (p : PState T) (acc : List T) : PState (List T) := do
  let v <- tryParse p
  match v with
  | none => return acc.reverse
  | some x =>
    ignore comma
    parseCommaListAux p (x :: acc)

-- We need to parse both () for shapes and {} for the outer metadata record
def parseCommaList {T : Type} (start : Char) (end_ : Char) (p : PState T) : PState (List T) := do
  consume start
  let xs <- parseCommaListAux p []
  consume end_
  return xs

partial def parseShape : PState Shape := do
  let xs <- parseCommaList '(' ')' parseToken
  return .fromList! (xs.map (fun x => x.toNat!))

-- major/minor/header-length
def parseHeader : PState (Nat × Nat) := do
  let s <- get
  let b := s.source
  if s.index != 0 then .error "Illegal start index"
  if b.size < 10 then .error "Buffer too small"
  if b[0]! != 0x93 then .error s!"Invalid first byte: {b[0]!}"
  if b[1]! != 'N'.toUInt8 then .error "Invalid second byte"
  if b[2]! != 'U'.toUInt8 then .error "Invalid third byte"
  if b[3]! != 'M'.toUInt8 then .error "Invalid fourth byte"
  if b[4]! != 'P'.toUInt8 then .error "Invalid fifth byte"
  if b[5]! != 'Y'.toUInt8 then .error "Invalid sixth byte"
  let major := b[6]!.toNat
  let minor := b[7]!.toNat
  -- Header length is 2 bytes, little-endian
  let headerLength := b[8]!.toNat + b[9]!.toNat * 256
  set { s with index := 10, headerEnd := 10 + headerLength }
  return (major, minor)

-- There are 3 fields, descr, fortran_order, and shape. We are not supposed to depend
-- on the ordering of the fields
def parseOneMetadata : PState Unit := do
  let id <- quoted parseToken
  colon
  if id == "descr" then
    let v <- quoted parseToken
    let d <- NumpyDtype.fromString v
    modify (fun s => { s with descr := some d })
  else if id == "fortran_order" then
    let v <- parseToken
    let b <- liftM (NumpyBool.fromString v)
    modify (fun s => { s with fortranOrder := b })
  else if id == "shape" then
    let shape <- parseShape
    modify (fun s => { s with shape := some shape })
  else .error s!"Unknown metadata key: {id}"

-- Slightly lame to call it 3 times, but there are always 3 elements, and
-- putting it in a list and searching for those values also seems annoying.
def parseMetadata : PState Unit := do
  consume '{'
  parseOneMetadata
  comma
  parseOneMetadata
  comma
  parseOneMetadata
  ignore comma
  consume '}'

def parseNumpyRepr : PState NumpyRepr := do
  let (major, minor) <- parseHeader
  parseMetadata
  let (s : ParsingState) <- get
  match s.descr, s.fortranOrder, s.shape with
  | some descr, some fortranOrder, some shape =>
    let numpyHeader := NumpyHeader.mk major minor descr fortranOrder shape
    let strides := shape.defaultStrides descr
    let repr := NumpyRepr.mk numpyHeader s.source s.headerEnd s.source.size strides
    return repr
  | _, _, _ => .error "Can't parse a metadata value"

def parse (buffer : ByteArray) : Except String NumpyRepr := do
  let init := ParsingState.mk buffer 0 0 none none none []
  resultExcept $ parseNumpyRepr.run init

def parseFile (path: System.FilePath) : IO (Except String NumpyRepr) := do
  let buffer <- IO.FS.readBinFile path
  return parse buffer

end Parse

end NumpyRepr
end TensorLib
