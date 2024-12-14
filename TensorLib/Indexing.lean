import TensorLib.Tensor
import TensorLib.Slice
import TensorLib.Npy

/-
Theorems to prove:

1. Basic slicing with more than one non-: entry in the slicing tuple,
   acts like repeated application of slicing using a single non-: entry,
   where the non-: entries are successively taken (with all other non-: entries replaced by :).
   Thus, x[ind1, ..., ind2,:] acts like x[ind1][..., ind2, :] under basic slicing.
-/

namespace TensorLib
namespace Index

-- Indexing utility
private def start (strides : Strides) (index : DimIndex) : Err ℕ :=
  match strides, index with
  | [], [] => .error "Can not index with empty list"
  | [], _ :: _ | _ :: _, [] => .error "Unequal lengths: strides and index"
  | stride :: strides, i :: index => (start strides index).map fun x => (x + i * stride).toNat

-- Get a single value from the tensor.
-- TODO: Replace get! with the Fin version. I tried this for a couple hours
-- and failed.
private def getBytes! (x : Tensor) (index : DimIndex) : Err (List UInt8) := do
  let gap <- start x.strides index
  let i := x.startIndex + gap
  let rec loop (n : ℕ) (acc : List UInt8) : List UInt8 :=
    match n with
    | 0 => acc
    | n + 1 => loop n (x.data.get! (i + n) :: acc)
  return loop x.itemsize []


-- This is a blind index into the array, disregarding the shape.
def getPosition [typ : Tensor.Element a] (x : Tensor) (position : ℕ) : Err a :=
  if typ.itemsize != x.itemsize then .error "byte size mismatch" else -- TODO: Lift this check out so we only do it once
  typ.fromByteArray x.data (x.startIndex + (position * typ.itemsize))

def setPosition [typ : Tensor.Element a] (x : Tensor) (n : ℕ) (v : a): Err Tensor :=
  let itemsize := typ.itemsize
  if itemsize != x.itemsize then .error "byte size mismatch" else -- TODO: Lift this check out so we only do it once
  let bytes := typ.toByteArray v
  let posn := n * itemsize
  .ok { x with data := bytes.copySlice 0 x.data posn itemsize true }

-- Since the DimIndex is independent of the dtype size, we need to recompute the strides
-- TODO: Would be better to not recompute this over and over. We should find a place to store
-- the 1-based default strides
def getDimIndex [Tensor.Element a] (x : Tensor) (index : DimIndex) : Err a :=
  let offset := Shape.dimIndexToOffset x.unitStrides index
  let posn := x.startIndex + offset
  if posn < 0 then .error s!"Illegal position: {posn}"
  else getPosition x posn.toNat

def setDimIndex [Tensor.Element a] (x : Tensor) (index : DimIndex) (v : a): Err Tensor :=
  let offset := Shape.dimIndexToOffset x.unitStrides index
  let posn := x.startIndex + offset
  if posn < 0 then .error s!"Illegal position: {posn}"
  else setPosition x posn.toNat v

-- Parsed form. We will simplify some of the redundancy here to yield an Index.Basic
-- In NumPy there may only be a single ellipsis present.
inductive NumpyItem where
| int (n : Int)
| slice (slice : Slice)
| ellipsis
| newaxis
deriving BEq

abbrev NumpyBasic := List NumpyItem

-- Slices do not subsume nat indexing. Slices always return an array
-- with the same shape as the input, while a nat index always reduces the shape by one dimension.
-- x
-- array([1, 2, 3, 4, 5, 6])
--
-- x[0], x[0:1:1]
-- (np.int64(1), array([1]))
--
-- x[0].shape, x[0:1:1].shape
-- ((), (1,))
inductive Item where
| nat (n : Nat)
| slice (slice : Slice)
| newaxis

abbrev Basic := List Item

structure Reshape where
  shape : Shape
  strides : Strides
  startPosition : ℕ
  -- TODO H : shape.length == strides.length

namespace Reshape

def checked (shape : Shape) (strides : Strides) (startPosition : ℕ) : Err Reshape :=
  if shape.length != strides.length then .error "Shape/Stride mismatch"
  else .ok { shape, strides, startPosition }

/-
TODO: reshapes can require a copy. For example, when we get the data out of order via
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

def applyBasic (reshape : Reshape) (basic : Basic) : Err Reshape :=
  match basic, reshape.shape, reshape.strides with
  | [], _, _ => .ok reshape
  | .newaxis :: basic, _, _ => do
    let rest <- applyBasic reshape basic
    return { rest with shape := 1 :: rest.shape, strides := 1 :: rest.strides }
  | .nat n :: basic, dim :: shape, stride :: strides =>
    if dim <= n
    then .error s!"Index out of range: {dim} <= {n}"
    else
      let startPosition := (reshape.startPosition + (n * stride)).toNat
      applyBasic (Reshape.mk shape strides startPosition) basic
  | .slice slice :: basic, dim :: shape, stride :: strides =>
    let start := slice.startOrDefault dim
    let stop := slice.stopOrDefault dim
    let step := slice.stepOrDefault
    if dim <= start then .error s!"Slice start out of range: {dim} <= {start}"
    else if stop.any (fun k => dim <= k) then .error s!"Slice stop out of range: {dim} <= {stop}"
    else
      let startPosition := (reshape.startPosition + (start * stride)).toNat
      do
        let rest <- applyBasic (Reshape.mk shape strides startPosition) basic
        return { rest with shape := slice.size dim :: rest.shape, strides := stride * step :: rest.strides }
  | _, _, _ => .error s!"Too many indices"

end Reshape

-- Return the basic index along with the shape that will result from the indexing
def simplifyNumpyBasic (items : NumpyBasic) (shape : Shape) : Err (Basic × Shape) := do
  if 1 < items.count .ellipsis then .error "Index can contain at most one ellipsis" else .ok ()
  if shape.length < items.length then .error "Too many indices" else .ok ()
  match items, shape with
  | [], shape => .ok ([], shape)
  | .int n :: items, dim :: shape => do
    -- Use a slice to do the finicky conversion to a position in the array
    let slice <- Slice.build (.some n) .none .none
    let n := slice.startOrDefault dim
    let (rest, shape) <- simplifyNumpyBasic items shape
    .ok (Item.nat n :: rest, shape)
  | .slice slice :: items, dim :: shape => do
    let (rest, shape) <- simplifyNumpyBasic items shape
    .ok (Item.slice slice :: rest, slice.size dim :: shape)
  | .newaxis :: items, dim :: shape => do
    let (rest, shape) <- simplifyNumpyBasic items (dim :: shape)
    .ok (Item.newaxis :: rest, 1 :: shape)
  | .ellipsis :: items, dim :: shape => do
    if items.length == 1 + shape.length then
      simplifyNumpyBasic items (dim :: shape)
    else -- there are fewer items than the shape
      let (index, shape) <- simplifyNumpyBasic (.ellipsis :: items) shape
      return (index, dim :: shape)
  | _, _ => .error "impossible"

end Index

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

private def testTensorElementBV (n : Nat) [Tensor.Element (BitVec n)] (dtype : String) : IO Bool := do
  let file <- saveNumpyArray s!"np.arange(20, dtype='{dtype}').reshape(5, 4)"
  let npy <- Npy.parseFile file
  let arr <- IO.ofExcept (Tensor.ofNpy npy)
  let _ <- debugPrintln file
  let _ <- debugPrintln arr
  let _ <- IO.FS.removeFile file
  let expected : List (BitVec n) := List.range 20
  let actual := do
    let rev <- Nat.foldM (fun i acc => (Index.getPosition arr i).map (fun i => i :: acc)) [] 20
    return rev.reverse
  let _ <- debugPrintln actual
  match actual with
  | .error _ => return false
  | .ok actual => return expected == actual

-- TODO: Asserting true/false here would be great
#eval testTensorElementBV 16 "uint16" -- expect true
#eval testTensorElementBV 32 "uint16" -- expect false
#eval testTensorElementBV 64 "uint16" -- expect false
#eval testTensorElementBV 16 "uint32" -- expect false
#eval testTensorElementBV 32 "uint32" -- expect true
#eval testTensorElementBV 64 "uint32" -- expect false
#eval testTensorElementBV 16 "uint64" -- expect false
#eval testTensorElementBV 32 "uint64" -- expect false
#eval testTensorElementBV 64 "uint64" -- expect true

end Test
end TensorLib
