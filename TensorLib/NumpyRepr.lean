import Init.System.IO
import Init.System.FilePath
import Mathlib.Tactic
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

abbrev Shape := List Nat

namespace Shape

def count (s: Shape): Nat := match s with
| [] => 0
| [x] => x
| x :: xs => x * count xs

-- def parse (s: String)

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
  startIndex : Nat -- Pointer to the first byte of ndarray data
  endIndex : Nat   -- Pointer to the byte after the last byte of ndarray data
  -- invariant: startIndex <= endIndex
  -- invariant: endIndex - startIndex = descr.bytes * shape.count
  -- invariant: header.endIndex <= |data|
  -- invariant: header.dataSize <= |data|
  deriving Repr

namespace NumpyRepr

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

abbrev PState (T : Type) := StateT ParsingState (Except String) T

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
  | .error _ => .ok (none, s)
  | .ok (x, s) => .ok (some x, s)

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

def parseShape : PState Shape := do
  let xs <- parseCommaList '(' ')' parseToken
  return xs.map (fun x => x.toNat!)

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
    let b <- NumpyBool.fromString v
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
    let repr := NumpyRepr.mk numpyHeader s.source s.headerEnd s.source.size
    return repr
  | _, _, _ => .error "Can't parse a metadata value"

def parse (buffer : ByteArray) : Except String NumpyRepr := do
  parseNumpyRepr.eval <| ParsingState.mk buffer 0 0 none none none []

def parseFile (path: System.FilePath) : IO (Except String NumpyRepr) := do
  let buffer <- IO.FS.readBinFile path
  return parse buffer

end Parse

end NumpyRepr
end TensorLib
