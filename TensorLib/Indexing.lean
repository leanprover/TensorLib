import TensorLib.Common
import TensorLib.Tensor
import TensorLib.Slice
import TensorLib.Npy

/-
Theorems to prove (taken from NumPy docs):

1. Basic slicing with more than one non-: entry in the slicing tuple,
   acts like repeated application of slicing using a single non-: entry,
   where the non-: entries are successively taken (with all other non-: entries replaced by :).
   Thus, x[ind1, ..., ind2,:] acts like x[ind1][..., ind2, :] under basic slicing.

2. ...TODO...
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
| slice (slice : Slice.Iter)

abbrev Basic := List Item

-- Return the basic index along with the shape that will result from the indexing
def toBasic (items : NumpyBasic) (shape : Shape) : Err Basic := do
  if 1 < items.count .ellipsis then .error "Index can contain at most one ellipsis"
  else if shape.length < items.length then .error "Too many indices"
  else
    let rec loop (items : NumpyBasic) (shape : Shape) : Err Basic := match items, shape with
    | [], _ => .ok []
    | .int n :: items, dim :: shape => do
      -- Use a slice to do the finicky conversion to a position in the array
      let slice := Slice.ofInt n
      let n := slice.startOrDefault dim
      let rest <- loop items shape
      .ok $ Item.nat n :: rest
    | .slice slice :: items, dim :: shape => do
      let rest <- loop items shape
      .ok $ Item.slice (Slice.Iter.make slice dim) :: rest
    | .newaxis :: items, dim :: shape => do
      let rest <- loop items (dim :: shape)
      .ok $ .slice (Slice.Iter.make Slice.all dim) :: rest
    | .ellipsis :: items, dim :: shape => do
      if items.length == 1 + shape.length then
        loop items (dim :: shape)
      else  -- there are fewer items than the shape
        loop (.ellipsis :: items) shape
    | _, _ => .error "impossible"
    loop items shape

  /-
   Iteration over a basic index is a little tricky. The new axes have been
   compiled away, but we still have constant indices and slice indices.
   These correspond roughly to constant counters and incrementing counters where
   we go lexicographically "upwards" during iteration (though the actual numbers
   may go down if we have negative steps.) We have to handle the following corner
   cases

   0. The index doesn't make sense. For example, the shape's dimension doesn't line
      up with the corresponding index. Say the shape's dimension is 2, but the iterator goes
      up to 7. As another error case, there may be more dimensions than indices.
   1. The index is empty on creation. E.g. if the shape is empty, or if the
      slice iterators are already all maxed out.
   2. We only have constant indices. This is an iterator of size 1, which we
      need to handle properly.

   All lists stored backwards, so make it hard to get at them
   Invariant: shape != []
   Invariant: shape.length == basic.length == next.length
   Invariant: each element of `basic` is in bounds for the corresponding dimension in `shape`.
              This is checked on the call to `make`.
  -/
  structure BasicIter where
    private mk ::
    private basic : Basic
    private curr : Option (List Nat) -- `curr` is `none` iff we are done
  deriving Inhabited

  namespace BasicIter

  def size (iter : BasicIter) : Nat :=
    if iter.curr.isNone then 0 else loop iter.basic 1
  where
    loop xs acc := match xs with
    | [] => acc
    | .nat _ :: xs => loop xs acc
    | .slice slice :: xs => loop xs (slice.size * acc)


  -- An iterator is compatible with a shape if all of the shape's dimensions are
  -- larger than the constant indices and equal to the iterator indices.
  private def compatibleWith (basic : Basic) (shape : Shape) : Bool := match basic, shape with
  | [], [] => true
  | .nat n :: basic, dim :: shape => n < dim && compatibleWith basic shape
  | .slice iter :: basic, dim :: shape => iter.dim == dim && compatibleWith basic shape
  | _, _ => false

  def hasNext (iter : BasicIter) : Bool := iter.curr.isSome

  /-
  Grabbing the first element is complicated by the fact that a slice index can be
  exhausted already, so we need to increment it. While calculating the first iteration, we
  will possibly rewrite the index to bump the exhausted sub-indices.

   `none` iff the iterator is empty. Does not handle error cases, which are handled by `make`.
  -/
  private def nextWithCarry (basic : Basic) (carry : Bool) : Option (List Nat × Basic) := match basic with
  | [] => if carry then .none else .some ([], [])
  | .nat n :: basic => do
    let (ns, basic) <- nextWithCarry basic carry
    return (n :: ns, .nat n :: basic)
  | .slice sliceIter :: basic => match sliceIter.next with
    | .none => do
      let sliceIter := sliceIter.reset
      let (ns, basic) <- nextWithCarry basic (carry := true)
      return (0 :: ns, .slice sliceIter :: basic)
    | .some (n, nextSliceIter) => do
      if !carry then
        let (ns, basic) <- nextWithCarry basic false
        return (n :: ns, .slice sliceIter :: basic)
      else
        if n < sliceIter.dim - 1 then
          let (ns, basic) <- nextWithCarry basic false
          return ((n+1) :: ns, .slice nextSliceIter :: basic)
        else
          let (ns, basic) <- nextWithCarry basic true
          let sliceIter := sliceIter.reset
          return (0 :: ns, .slice sliceIter :: basic)

  def next (iter : BasicIter) : Option (List Nat × BasicIter) :=
    match nextWithCarry iter.basic false with
    | .none => .none
    | .some (ns, basic) => .some (ns, { iter with basic })

  def make (shape : Shape) (basic : Basic) : Err BasicIter :=
    if !(compatibleWith basic shape) then .error "shape/index mismatch" else
    let basic := basic.reverse
    .ok $ match nextWithCarry basic false with
    | .none => BasicIter.mk basic .none
    | .some (ns, basic) => BasicIter.mk basic (.some ns)

instance [Monad m] : ForIn m BasicIter (List Nat) where
  forIn {α} [Monad m] (iter : BasicIter) (x : α) (f : List Nat -> α -> m (ForInStep α)) : m α := do
    let mut iter : BasicIter := iter
    let mut res := x
    for _ in [0:iter.size] do
      match iter.next with
      | .none => break
      | .some (ns, iter') =>
        iter := iter'
        match <- f ns res with
        | .yield k
        | .done k => res := k
    return res

private def toList (iter : BasicIter) : List (List Nat) := Id.run do
  let mut res := []
  for xs in iter do
    res := xs :: res
  return res.reverse

-- Testing
private def numpyBasicToList (shape : Shape) (basic : NumpyBasic) : Option (List (List Nat)) := do
  let basic <- (toBasic basic shape).toOption
  let iter <- (make shape basic).toOption
  iter.toList

#guard numpyBasicToList [] [] == .some [[]]
#guard numpyBasicToList [1] [.int 0] == .some [[0]]
#guard numpyBasicToList [2] [.int 0] == .some [[0]]
#guard numpyBasicToList [2] [.int 1] == .some [[1]]
#guard (numpyBasicToList [2] [.int 2]) == .none
#guard (numpyBasicToList [2] [.int (-1)]) == some [[1]]
#guard (numpyBasicToList [2] [.int (-2)]) == some [[0]]

end BasicIter
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
