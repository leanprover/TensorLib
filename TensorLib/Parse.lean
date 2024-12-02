import Init.System.IO
import TensorLib.NumpyRepr

--! Parse a .npy file
-- Inputs to TenCert tend to be arrays stored in .npy files. This module parses and saves the npy file format.
-- https://numpy.org/devdocs/reference/generated/numpy.lib.format.html

namespace TensorLib
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

abbrev PState (T : Type) := EStateM String ParsingState T

instance : MonadLiftT Err PState where
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
private def numpyBool (s: String): Err Bool := match s with
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
  return xs.map (fun x => x.toNat!)

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
    let d <- NumpyDtype.fromString v
    modify (fun s => { s with descr := some d })
  else if id == "fortran_order" then
    let v <- parseToken
    let b <- liftM (numpyBool v)
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

private def parseNumpyRepr : PState NumpyRepr := do
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

def parse (buffer : ByteArray) : Err NumpyRepr := do
  let init := ParsingState.mk buffer 0 0 none none none []
  resultExcept $ parseNumpyRepr.run init

def parseFile (path: System.FilePath) : IO (Err NumpyRepr) := do
  let buffer <- IO.FS.readBinFile path
  return parse buffer

end Parse

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
private def toByteArray! (repr : NumpyRepr) : ByteArray :=
  let a := ByteArray.empty.push 0x93
  let a := pushString a "NUMPY"
  let a := pushList a [repr.header.major, repr.header.minor]
  let a := (a.push 0).push 0 -- index 8, 9. We will clobber this with the header size in a moment
  if a.size != 10 then panic s!"Bad header size: {a.size}, should be 9" else
  let a := pushStrings a [
    "{'descr': '",
    repr.header.descr.toString,
    "', 'fortran_order': ",
    boolString repr.header.fortranOrder,
    ", 'shape': (",
  ]
  let shape := repr.header.shape
  let a := if H : shape.isEmpty then a else
    let ok : shape ≠ [] := by
      simp at H
      exact H
    let a := pushString a (toString (shape.head ok))
    repr.header.shape.tail.foldl (fun a d => pushString (pushString a ", ") (toString d)) a
  let a := pushString a "), }"
  -- We need the header to be aligned
  let padding := 64 - (a.size % 64) - 1 -- -1 for the terminal '\n'
  let a := pushList a (List.replicate padding 0x20)
  let a := a.push 0x0a -- '\n'
    -- header size is little-endian
  let (low, hi) := headerSizeToBytes (a.size - 10)
  let a := a.set! 8 low
  let a := a.set! 9 hi
  let data' := repr.data.copySlice repr.startIndex ByteArray.empty 0 repr.nbytes
  a.append data'

def save! (repr : NumpyRepr) (file : System.FilePath) : IO Unit :=
  IO.FS.writeBinFile file repr.toByteArray!

end Save

section Test

private def DEBUG := false
private def debugPrint {a : Type} [Repr a] (s : a) : IO Unit := if DEBUG then IO.print (Std.Format.pretty (repr s)) else return ()
private def debugPrintln {a : Type} [Repr a] (s : a) : IO Unit := do
  debugPrint s
  if DEBUG then IO.print "\n" else return ()

-- Caller must remove the temp file
private def saveNumpyArray (expr : String) : IO System.FilePath := do
  let (_root_, file) <- IO.FS.createTempFile
  let expr := s!"import numpy as np; x = {expr}; np.save('{file}', x)"
  let output <- IO.Process.output { cmd := "/usr/bin/env", args := ["python3", "-c", expr].toArray }
  let _ <- debugPrintln output.stdout
  let _ <- debugPrintln output.stderr
  -- `np.save` appends `.npy` to the file
  return file.addExtension "npy"

private def testTensorElementBV {n : Nat} (w : TensorElement (BitVec n)) (dtype : String) : IO Bool := do
  let file <- saveNumpyArray s!"np.arange(20, dtype='{dtype}').reshape(5, 4)"
  let arr <- Parse.parseFile file
  let _ <- debugPrintln file
  let _ <- debugPrintln arr
  let _ <- IO.FS.removeFile file
  let expected : List (BitVec n) := List.range 20
  let actual := arr.bind (fun arr => do
    let rev <- Nat.foldM (fun i acc => (Index.rawIndex w arr i).map (fun i => i :: acc)) [] 20
    return rev.reverse
  )
  let _ <- debugPrintln actual
  match actual with
  | .error _ => return false
  | .ok actual => return expected == actual

-- TODO: Asserting true/false here would be great
#eval testTensorElementBV TensorElement.BV16 "uint16" -- expect true
#eval testTensorElementBV TensorElement.BV32 "uint16" -- expect false
#eval testTensorElementBV TensorElement.BV64 "uint16" -- expect false
#eval testTensorElementBV TensorElement.BV16 "uint32" -- expect false
#eval testTensorElementBV TensorElement.BV32 "uint32" -- expect true
#eval testTensorElementBV TensorElement.BV64 "uint32" -- expect false
#eval testTensorElementBV TensorElement.BV16 "uint64" -- expect false
#eval testTensorElementBV TensorElement.BV32 "uint64" -- expect false
#eval testTensorElementBV TensorElement.BV64 "uint64" -- expect true

end Test

end NumpyRepr
end TensorLib
