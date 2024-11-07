import Init.System.IO
import Init.System.FilePath
import Mathlib.Tactic
import TensorLib.NonEmptyList
import TensorLib.TensorElement

/-
https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
https://github.com/numpy/numpy/blob/067cb067cb17a20422e51da908920a4fbb3ab851/doc/neps/nep-0001-npy-format.rst
-/

namespace TensorLib

-- The subset of types NumPy supports that we care about
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

def bytes (x: NumpyDtype): Nat := match x with
| float64 | int64 | uint64 => 8
| float32 | int32 | uint32 => 4
| float16 | int16 | uint16 => 2
| bool | int8 | uint8 => 1

-- Disk formats found through experimentation. Not sure why there are
-- both '<' and '|' as prefixes
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

def NumpyBool.fromString (s: String): Except String Bool := match s with
| "True" => .ok true
| "False" => .ok false
| _ => .error s!"Can't parse {s} as Bool"

abbrev Shape := NonEmptyList Nat
abbrev Strides := NonEmptyList Nat

namespace Shape

def count (s : Shape) : Nat := match s with
| .mk x xs => x * xs.foldl (fun x y => x * y) 1

def defaultStrides (dtype : NumpyDtype) (s : Shape) : Strides :=
  let s := s.reverse
  let bytes := dtype.bytes
  let rec loop (xs : List ℕ) (lastShape lastDimSize : ℕ): List ℕ := match xs with
  | [] => []
  | d :: ds =>
    let rest := loop ds (lastShape * lastDimSize) d
    lastShape * lastDimSize :: rest
  let res := NonEmptyList.mk dtype.bytes (loop s.tl bytes s.hd)
  res.reverse

#eval defaultStrides NumpyDtype.uint32 { hd := 2, tl := [] }
#eval defaultStrides NumpyDtype.uint32 { hd := 2, tl := [3] }
#eval defaultStrides NumpyDtype.uint32 { hd := 2, tl := [3, 5, 7] }

end Shape

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

def dataSize (header : NumpyHeader): Nat := header.descr.bytes * header.shape.count

end NumpyHeader

instance ByteArrayRepr : Repr ByteArray where
  reprPrec x _ :=
    let s := toString x.size
    s!"ByteArray of size {s}"

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

def shape (x : NumpyRepr) : Shape := x.header.shape

def ndim (x : NumpyRepr) : ℕ := x.shape.length

def size (x : NumpyRepr) : ℕ := x.shape.count

def itemsize (x : NumpyRepr) : ℕ := x.header.descr.bytes

def nbytes (x : NumpyRepr) : ℕ := x.itemsize * x.size

section Shape

def M a := Except String a
variable [Monad M]

def reshape (x : NumpyRepr) (shape : Shape) : M NumpyRepr := do
  if x.shape.count == shape.count then
    return { x with header.shape := shape }
  else
    .error "Reshaping must have the same number of implied elements"

end Shape

def replicateAux {α : Type} (n : Nat) (x : α) (acc : List α) : List α := match n with
| 0 => acc
| n + 1 => replicateAux n x (x :: acc)

theorem replicateAuxLength (n : Nat) (x : α) (acc : List α): (replicateAux n x acc).length = n + acc.length := by
  revert acc
  induction n
  . unfold replicateAux
    simp
  . unfold replicateAux
    intros acc
    rename_i n H
    rw [H (x :: acc)]
    simp
    linarith

def replicate {α : Type} (n : Nat) (x : α) : List α :=
  replicateAux n x []

theorem replicateLength (n : Nat) (x : α) : (replicate n x).length = n := by
  exact (replicateAuxLength n x [])

structure Broadcast where
  left : Shape
  right : Shape
  deriving BEq, Repr

section Broadcast

-- In broadcasting, we first extend the shorter array by prefixing 1s.
-- NKI semantics currently suffixes 1s in some cases, so be explicit about
-- the naming
def oneExtendPrefix (b : Broadcast) : Broadcast :=
  let n1 := b.left.length
  let n2 := b.right.length
  if n1 <= n2
  then { b with left := replicate (n2 - n1) 1 ++ b.left }
  else { b with right := replicate (n1 - n2) 1 ++ b.right }

theorem oneExtendPrefixLength (b : Broadcast) :
  let b' := oneExtendPrefix b
  b'.left.length = b'.right.length := by
  cases b
  rename_i left right
  unfold oneExtendPrefix
  simp
  by_cases H : left.length <= right.length
  . simp [H]
    rw [NonEmptyList.hAppendListLLength, replicateLength, Nat.sub_add_cancel]
    exact H
  . simp [H]
    rw [NonEmptyList.hAppendListLLength, replicateLength, Nat.sub_add_cancel]
    linarith

def matchPairs (b : Broadcast) : Option Shape :=
  if b.left.length != b.right.length then none else
  let f xy := match xy with
    | (x, y) =>
      if x == y then some x
      else if x == 1 then some y
      else if y == 1 then some x
      else none
  traverse f (NonEmptyList.zip b.left b.right)

def broadcast (b : Broadcast) : Option Shape := matchPairs (oneExtendPrefix b)

def canBroadcast (b : Broadcast) : Bool := (broadcast b).isSome

#eval matchPairs (Broadcast.mk { hd := 1, tl := [2, 3] } { hd := 7, tl := [2, 1] })
#eval broadcast (Broadcast.mk { hd := 1, tl := [2, 3] } { hd := 7, tl := [7, 9, 2, 1] })

-- todo: add plausible properties when property-based testing settles down in Lean-land
#guard
 let x1 := NonEmptyList.fromList! [1,2,3]
 let x2 := NonEmptyList.fromList! [2,3]
 let b1 := Broadcast.mk x1 x1
 let b2 := Broadcast.mk x1 x2
 oneExtendPrefix b1 == b1 &&
 broadcast b2 == broadcast b1 &&
 broadcast b2 == some (NonEmptyList.mk 1 [2, 3])

end Broadcast

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
  return NonEmptyList.fromList! (xs.map (fun x => x.toNat!))

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
