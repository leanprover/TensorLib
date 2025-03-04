/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/

import TensorLib.Common
import TensorLib.Dtype

/-!
We largely duplicate the NumPy representation of tensors.

The binary format is described here: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
and here: https://github.com/numpy/numpy/blob/067cb067cb17a20422e51da908920a4fbb3ab851/doc/neps/nep-0001-npy-format.rst

In addition to being an efficient representation, this allows us to directly parse .npy input files.

NumPy stores the data in either row major (C) or column major (Fortran) order
Indexing and other operations take the data ordering into account. For example,
transposing a 2D matrix can be accomplished by changing the ordering and updating
the shapes and strides. Since Fortran order is rare, and it seems we can do everything
we want to do for the time being with shape and strides, we ignore the data aorder
and fail if we see a fortran array on disk. If you need to read a Fortran array in
.npy format, you can do

# x = np.arange(10).reshape(2, 5)
# np.save("/tmp/foo.npy", x.T)
# np.save('/tmp/bar.npy', np.asarray(x.T, order='C'))
# hexdump -C /tmp/foo.npy
...'fortran_order': True... 0 5 1 6 ...
# hexdump -C /tmp/bar.npy
...'fortran_order': False...0 1 2 3
-/

namespace TensorLib
namespace Npy

/-!
NumPy has an extra constructor for byte orders "native", so we just
copy the other fields.
-/
inductive ByteOrder where
| native -- Leaves the ordering up to the machine reading the data
| littleEndian
| bigEndian
| notApplicable -- single byte types
deriving BEq, Repr, Inhabited

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

end ByteOrder

structure Dtype where
  name : TensorLib.Dtype
  order : ByteOrder
deriving BEq, Repr, Inhabited

namespace Dtype

/-!
Parse a numpy dtype. The first character represents the
byte order: https://numpy.org/doc/2.1/reference/generated/numpy.dtype.byteorder.html
-/
def dtypeNameFromNpyString (s : String) : Err TensorLib.Dtype := match s with
| "b1" => .ok .bool
| "i1" => .ok .int8
| "i2" => .ok .int16
| "i4" => .ok .int32
| "i8" => .ok .int64
| "u1" => .ok .uint8
| "u2" => .ok .uint16
| "u4" => .ok .uint32
| "u8" => .ok .uint64
| "f4" => .ok .float32
| "f8" => .ok .float64
| _ => .error s!"Can't parse {s} as a dtype"

def dtypeNameToNpyString (t : TensorLib.Dtype) : String := match t with
| .bool => "b1"
| .int8 => "i1"
| .int16 => "i2"
| .int32 => "i4"
| .int64 => "i8"
| .uint8 => "u1"
| .uint16 => "u2"
| .uint32 => "u3"
| .uint64 => "u4"
| .float32 => "f4"
| .float64 => "f8"

def fromNpyString (s : String) : Err Dtype :=
  if s.length == 0 then .error "Empty dtype string" else
  do
    let order <- ByteOrder.fromChar (s.get 0)
    let name <- dtypeNameFromNpyString (s.drop 1)
    return { name, order }

def toNpyString (t : Dtype) : String := t.order.toChar.toString.append (dtypeNameToNpyString t.name)

def itemsize (t : Dtype) := t.name.itemsize

end Dtype

/-!
This is the header of the on-disk Numpy format, typically with the .npy file extension.

https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#format-version-1-0
-/
structure Header where
  major : UInt8 := 1
  minor : UInt8 := 0
  descr : Dtype
  shape : Shape
  deriving Repr, Inhabited

namespace Header

/-!
A npy binary file has a header, some padding, then the data. This method computes the
size of the data portion of the file.
-/
def dataSize (header : Header): Nat := header.descr.itemsize * header.shape.count

end Header

structure Ndarray where
  header : Header
  data : ByteArray
  startIndex : Nat -- First byte of non-header data
  deriving Repr, Inhabited

namespace Ndarray

def nbytes (x : Ndarray) : Nat := x.header.descr.itemsize * x.header.shape.count

def dtype (arr : Ndarray) : Dtype := arr.header.descr

def itemsize (arr : Ndarray) : Nat := arr.dtype.itemsize

def order (arr : Ndarray) : ByteOrder := arr.dtype.order

end Ndarray

section Parse

private structure ParsingState where
  source : ByteArray    -- Source data being parsed
  index : Nat           -- Index into source data
  headerEnd : Nat
  descr : Option Dtype
  fortranOrder : Option Bool
  shape : Option Shape
  debug : List String
  deriving Repr

private abbrev PState (T : Type) := EStateM String ParsingState T

private instance : MonadLiftT Err PState where
  monadLift x := match x with
  | .ok x => .ok x
  | .error x => .error x

-- Except is nicer sometimes
private def resultExcept {σ : Type} (x : EStateM.Result String σ a) : Err a := match x with
| .ok x _ => .ok x
| .error x _ => .error x

variable {T : Type}

private def debug (msg : String) : PState Unit := do
  modify (fun s => { s with debug := msg :: s.debug })

-- Numpy disk format uses python convention for bools; True/False
private def npyBool (s: String): Err Bool := match s with
| "True" => .ok true
| "False" => .ok false
| _ => .error s!"Can't parse {s} as Bool"

-- Whitespace is under-specified in the docs linked above, but I think
-- Lean's isWhitespace will be fine here. I have never seen a non 0x20
-- space character in my experiments.
private def whitespace : PState Unit := do
  let s <- get
  let mut count := 0
  for i in [s.index : s.headerEnd] do
    let c := Char.ofUInt8 s.source[i]!
    if c.isWhitespace then
      count := count + 1
    else
      break
  set { s with index := s.index + count }

private def tryParse (p : PState T) : PState (Option T) := fun s =>
  match p s with
  | .error _ s => .ok none s
  | .ok x s => .ok (some x) s

private def ignore (p : PState T) : PState Unit := do
  let _ <- tryParse p
  return ()

-- Does some heavy lifting, parsing both normal ids (map keys) and weird
-- values like the dtypes
private def parseToken : PState String := do
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

private def consume (c: Char) : PState Unit := do
  whitespace
  let s <- get
  let i := s.index
  if s.source[i]! == c.toUInt8 then
    set { s with index := i + 1 }
  else
    .error s!"Can't consume {c}. So far we have {repr s}"

private def quote : PState Unit := consume '\''
private def colon : PState Unit := consume ':'
private def comma : PState Unit := consume ','

private def quoted (p : PState T) : PState T := do
  quote
  let x <- p
  quote
  return x

private partial def parseCommaListAux (p : PState T) (acc : List T) : PState (List T) := do
  let v <- tryParse p
  match v with
  | none => return acc.reverse
  | some x =>
    ignore comma
    parseCommaListAux p (x :: acc)

private def parseCommaList {T : Type} (start : Char) (end_ : Char) (p : PState T) : PState (List T) := do
  consume start
  let xs <- parseCommaListAux p []
  consume end_
  return xs

private partial def parseShape : PState Shape := do
  let xs <- parseCommaList '(' ')' parseToken
  return Shape.mk $ xs.map (fun x => x.toNat!)

-- major/minor/header-length
private def parseHeader : PState (UInt8 × UInt8) := do
  let s <- get
  let b := s.source
  if s.index != 0 then .error "Illegal start index"
  if b.size < 10 then .error s!"Buffer too small: {b.size}"
  if b[0]! != 0x93 then .error s!"Invalid first byte: {b[0]!}"
  if b[1]! != 'N'.toUInt8 then .error "Invalid second byte"
  if b[2]! != 'U'.toUInt8 then .error "Invalid third byte"
  if b[3]! != 'M'.toUInt8 then .error "Invalid fourth byte"
  if b[4]! != 'P'.toUInt8 then .error "Invalid fifth byte"
  if b[5]! != 'Y'.toUInt8 then .error "Invalid sixth byte"
  let major := b[6]!
  let minor := b[7]!
  -- Header length is 2 bytes, little-endian
  let headerLength := b[8]!.toNat + b[9]!.toNat * 256
  set { s with index := 10, headerEnd := 10 + headerLength }
  return (major, minor)

-- There are 3 fields, descr, fortran_order, and shape. We are not supposed to depend
-- on the ordering of the fields
private def parseOneMetadata : PState Unit := do
  let id <- quoted parseToken
  colon
  if id == "descr" then
    let v <- quoted parseToken
    let d <- Dtype.fromNpyString v
    modify (fun s => { s with descr := some d })
  else if id == "fortran_order" then
    let v <- parseToken
    let b <- liftM (npyBool v)
    modify (fun s => { s with fortranOrder := b })
  else if id == "shape" then
    let shape <- parseShape
    modify (fun s => { s with shape := some shape })
  else .error s!"Unknown metadata key: {id}"

-- Slightly lame to call it 3 times, but there are always 3 elements, and
-- putting it in a list and searching for those values also seems annoying.
private def parseMetadata : PState Unit := do
  consume '{'
  parseOneMetadata
  comma
  parseOneMetadata
  comma
  parseOneMetadata
  ignore comma
  consume '}'

private def parseNpyRepr : PState Ndarray := do
  let (major, minor) <- parseHeader
  parseMetadata
  let (s : ParsingState) <- get
  match s.descr, s.shape with
  | some descr, some shape =>
    match s.fortranOrder with
    | .none | .some false => .ok () -- C data order, which is what we support
    | .some true => .error "Fortran data ordering not supported"
    let header := Header.mk major minor descr shape
    let repr := Ndarray.mk header s.source s.headerEnd
    return repr
  | _, _ => .error "Can't parse a metadata value"

def parse (buffer : ByteArray) : Err Ndarray := do
  let init := ParsingState.mk buffer 0 0 none none none []
  resultExcept $ parseNpyRepr.run init

def parseFile (path: System.FilePath) : IO Ndarray := do
  let buffer <- IO.FS.readBinFile path
  IO.ofExcept (parse buffer)

end Parse

/-
Write ndarray to disk in .npy format

Note that because the stride info is not saved, we may need a copy to save the array.
For example, saving `np.arange(6)[::-1]` results in the bytes stored as
05 04 03 02 01 00, which is a copy of the data, which is stored in the reverse order.
-/
section Save

private def pushList (a : ByteArray) (xs : List UInt8) : ByteArray := a.append xs.toByteArray

private def pushString (a : ByteArray) (xs : String) : ByteArray := a.append xs.toUTF8

private def pushStrings (a : ByteArray) (xs : List String) : ByteArray := xs.foldl pushString a

private def boolString (b : Bool) : String := if b then "True" else "False"

private def headerSizeToBytes (n : Nat) : UInt8 × UInt8 :=
  let v := n.toUInt16
  (v.toUInt8, (v >>> 8).toUInt8)

private def next64 (n : Nat) : Nat := 64 - (n % 64)

-- Can we do this with local mutation?
private def Ndarray.toByteArray! (arr : Ndarray) : ByteArray :=
  let a := ByteArray.empty.push 0x93
  let a := pushString a "NUMPY"
  let a := pushList a [arr.header.major, arr.header.minor]
  let a := (a.push 0).push 0 -- index 8, 9. We will clobber this with the header size in a moment
  if a.size != 10 then impossible s!"Bad header size: {a.size}, should be 9" else
  let order := false -- We only support C order
  let a := pushStrings a [
    "{'descr': '",
    arr.header.descr.toNpyString,
    "', 'fortran_order': ",
    boolString order,
    ", 'shape': (",
  ]
  let shape := arr.header.shape
  let a := if H : shape.val.isEmpty then a else
    let ok : shape.val ≠ [] := by
      simp at H
      exact H
    let a := pushString a (toString (shape.val.head ok))
    arr.header.shape.val.tail.foldl (fun a d => pushString (pushString a ", ") (toString d)) a
  let a := pushString a "), }"
  -- We need the header to be aligned
  let padding := 64 - (a.size % 64) - 1 -- -1 for the terminal '\n'
  let a := pushList a (List.replicate padding 0x20)
  let a := a.push 0x0a -- '\n'
  -- header size is little-endian
  let (low, hi) := headerSizeToBytes (a.size - 10)
  let a := a.set! 8 low
  let a := a.set! 9 hi
  -- TODO: Do a copy if the data is not forward and contiguous
  let data' := arr.data.copySlice arr.startIndex ByteArray.empty 0 arr.nbytes
  a.append data'

end Save

def Ndarray.save! (arr : Ndarray) (file : System.FilePath) : IO Unit :=
  IO.FS.writeBinFile file arr.toByteArray!

end Npy
end TensorLib
