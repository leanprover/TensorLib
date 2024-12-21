import TensorLib.Common
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

-- Return the basic index along with the shape that will result from the indexing
def toBasic (items : NumpyBasic) (shape : Shape) : Err (Basic × Shape) := do
  if 1 < items.count .ellipsis then .error "Index can contain at most one ellipsis" else .ok ()
  if shape.length < items.length then .error "Too many indices" else .ok ()
  match items, shape with
  | [], shape => .ok ([], shape)
  | .int n :: items, dim :: shape => do
    -- Use a slice to do the finicky conversion to a position in the array
    let slice <- Slice.build (.some n) .none .none
    let n := slice.startOrDefault dim
    let (rest, shape) <- toBasic items shape
     -- Nat indexes drop the axis
    .ok (Item.nat n :: rest, shape)
  | .slice slice :: items, dim :: shape => do
    let (rest, shape) <- toBasic items shape
    .ok (Item.slice slice :: rest, slice.size dim :: shape)
  | .newaxis :: items, dim :: shape => do
    let (rest, shape) <- toBasic items (dim :: shape)
    .ok (Item.newaxis :: rest, 1 :: shape)
  | .ellipsis :: items, dim :: shape => do
    if items.length == 1 + shape.length then
      toBasic items (dim :: shape)
    else -- there are fewer items than the shape
      let (index, shape) <- toBasic (.ellipsis :: items) shape
      return (index, dim :: shape)
  | _, _ => .error "impossible"

  -- All lists stored backwards, so make it hard to get at them
  structure BasicIter where
    private mk ::
    private shape : Shape -- Shape of the input array
    private basic : Basic -- Basic index
    private curr : List Nat -- Current position in the iteration

  namespace BasicIter

  private def init (shape : Shape) (basic : Basic) : List Nat := match shape, basic with
  | _ :: shape, .nat n :: basic => n :: init shape basic
  | dim :: shape, .slice slice :: basic  => slice.startOrDefault dim :: init shape basic
  | _ :: shape, .newaxis :: basic => 0 :: init shape basic
  | _, _ => []

  def make (shape : Shape) (basic : Basic) : BasicIter :=
    let s := shape.reverse
    let b := basic.reverse
    let c := init s b
    BasicIter.mk s b c

  def next (iter : BasicIter) : List Nat × BasicIter :=
    let rec loop (curr : List Nat) (basic : Basic) : List Nat := match curr, basic with
    | _ :: curr, Item.nat m :: basic => m :: next
    | _, _ => []
    let curr := loop iter.curr iter.basic
    { iter with curr }
  end BasicIter














structure Reshape where
  shape : Shape
  strides : Strides
  startPosition : ℕ
  -- TODO H : shape.length == strides.length
namespace Reshape

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
    let rev <- Nat.foldM (fun i acc => (Tensor.Element.getPosition arr i).map (fun i => i :: acc)) [] 20
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
