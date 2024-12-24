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
deriving Repr

abbrev Basic := List Item

def toBasic (items : NumpyBasic) (shape : Shape) : Err Basic := do
  if 1 < items.count .ellipsis then .error "Index can contain at most one ellipsis"
  else if shape.length < items.length then .error "Too many indices"
  else
    let rec loop (items : NumpyBasic) (shape : Shape) : Err Basic := match items, shape with
    | [], _ => .ok []
    | .int n :: items, dim :: shape => do
      -- constant indices throw on overflow/underflow
      if n <= -dim || dim <= n then .error "Constant index out of bounds" else
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

We store the slice iterators (backwards) as counters. We find the next index by incrementing
from left to right. If one of the slices is maxxed out, we reset it and carry, like in the
(backwards) usual digit-wise nat addition algorithm. We remember whether we've returned
the last element with `done`. We can't just have the counters because then we couldn't distinguish between the
case where we still have to return the last element, or if we've already returned it and
need to stop iteration. For example, in a binary counter, if we start at 000..0 and get
to 111...1, have we returned the 111...1 or not? Or perhaps more obviously, if all indices are `.nat k`, have
we returned the single element or not?

List fields are stored backwards for ease of iteration, so we make it harder to get at them
and use them incorrectly.

Invariant: `shape != []`
Invariant: `shape.length == basic.length == curr.length`
Invariant: each element of `basic` is in bounds for the corresponding dimension in `shape`.
           This is checked on the call to `make`. In particular, the slice iterators are reset
           before they reach the dimension.
-/
structure BasicIter where
  private mk ::
  private basic : Basic
  private done : Bool
deriving Inhabited, Repr

namespace BasicIter

def hasNext (iter : BasicIter) : Bool := !iter.done

-- How many iterations total, based purely on the shapes, not where we are in the iteration
def size (iter : BasicIter) : Nat := Id.run do
  let mut res := 1
  for i in iter.basic do
    match i with
      | .nat _ => ()
      | .slice slice => res := res * slice.size
  return res

-- An iterator is compatible with a shape if all of the shape's dimensions are
-- larger than the constant indices and equal to the iterator indices.
private def compatibleWith (iter : BasicIter) (shape : Shape) : Bool :=
  let rec loop := fun
  | [], [] => true
  | .nat n :: basic, dim :: shape => n < dim && loop basic shape
  | .slice iter :: basic, dim :: shape => iter.dim == dim && loop basic shape
  | _, _ => false
  loop iter.basic shape

-- The (reversed) current index iteration (not yet returned)
private def current (basic : Basic) : List Nat := match basic with
| [] => []
| .nat n :: basic => n :: current basic
| .slice slice :: basic => slice.peek.getD 0 :: current basic -- The peek is .some by invariant

-- Returns the current element (that hasn't yet been returned), and the next iterator.
private def nextWithCarry (basic : Basic) (carry : Bool) : List Nat × Option Basic :=
  match basic with
  | [] => ([], if carry then none else some [])
  | .nat n :: basic =>
    -- constant indices don't increment, so we need to keep looking for a slice iterator to increment
    let (ns, basic) := nextWithCarry basic carry
    (n :: ns, basic.map fun b => .nat n :: b)
  | .slice sliceIter :: basic =>
    match sliceIter.next with
    -- We already reset the iterator once we return the max element
    | .none => panic "Invariant violation"
    | .some (n, nextSliceIter) =>
      if nextSliceIter.hasNext
      then (n :: current basic, .slice nextSliceIter :: basic)
      else
        let iter := sliceIter.reset
        let (ns, basic) := nextWithCarry basic (carry := true)
        let basic := basic.map fun b => .slice iter :: b
        (n :: ns, basic)

def next (iter : BasicIter) : Option (List Nat × BasicIter) :=
  if iter.done then none else
  let (ns, basic) := nextWithCarry iter.basic false
  let ns := ns.reverse
  let basic := match basic with
  | none => { iter with done := true } -- All slice iterators are maxxed out in this case. Should we check this?
  | some basic => { iter with basic }
  some (ns, basic)

def make (shape : Shape) (basic : Basic) : Err BasicIter :=
  let iter := BasicIter.mk basic.reverse false
  if !(compatibleWith iter shape.reverse) then .error "shape/index mismatch" else
  .ok iter

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

section Test

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

#guard numpyBasicToList [] [] == some [[]]
#guard numpyBasicToList [1] [.int 0] == some [[0]]
#guard numpyBasicToList [2] [.int 0] == some [[0]]
#guard numpyBasicToList [2] [.int 1] == some [[1]]
#guard (numpyBasicToList [2] [.int 2]) == none
#guard (numpyBasicToList [2] [.int (-1)]) == some [[1]]
#guard (numpyBasicToList [2] [.int (-3)]) == none
#guard (numpyBasicToList [4] [.slice Slice.all]) == some [[0], [1], [2], [3]]
#guard (numpyBasicToList [4] [.slice $ Slice.build! .none .none (.some 2)]) == some [[0], [2]]
#guard (numpyBasicToList [4] [.slice $ Slice.build! (.some (-1)) .none (.some (-2))]) == some [[3], [1]]
#guard (numpyBasicToList [2, 2] [.int 5]) == none
#guard (numpyBasicToList [2, 2] [.int 0]) == none
#guard (numpyBasicToList [2, 2] [.int 0, .int 0]) == some [[0, 0]]
#guard (numpyBasicToList [2, 2] [.int 0, .int 1]) == some [[0, 1]]
#guard (numpyBasicToList [2, 2] [.int 0, .int 2]) == none
#guard (numpyBasicToList [3, 3] [.slice Slice.all, .int 2]) == some [[0, 2], [1, 2], [2, 2]]
#guard (numpyBasicToList [3, 3] [.int 2, .slice Slice.all]) == some [[2, 0], [2, 1], [2, 2]]
#guard (numpyBasicToList [2, 2] [.slice Slice.all, .slice Slice.all]) == some [[0, 0], [0, 1], [1, 0], [1, 1]]
#guard (numpyBasicToList [2, 2] [.slice (Slice.build! .none .none (.some (-1))), .slice Slice.all]) == some [[1, 0], [1, 1], [0, 0], [0, 1]]
#guard (numpyBasicToList [4, 2] [.slice (Slice.build! .none .none (.some (-2))), .slice Slice.all]) == some [[3, 0], [3, 1], [1, 0], [1, 1]]

-- Commented for easier debugging. Remove some day
-- #eval do
--   let shape := [4, 2]
--   let basic := get! $ toBasic [.slice (Slice.build! .none .none (.some (-2))), .slice Slice.all] shape
--   let iter0 := (get! $ make shape basic)
--   let (ns0, iter1) <- iter0.next
--   let (ns1, iter2) <- iter1.next
--   let (ns2, iter3) <- iter2.next
--   -- let (ns4, iter4) <- iter3.next
--   -- let (ns5, iter5) <- iter4.next
--   -- let (ns6, iter6) <- iter5.next
--   -- let (ns7, iter7) <- iter6.next
--   -- let (ns8, iter8) <- iter7.next
--   -- let (ns9, iter9) <- iter8.next
--   return (basic, iter0, ns0, iter1, ns1, iter2, ns2, iter3) -- , ns4, iter4) -- , ns5, iter5, ns6, iter6, ns7, iter7, ns8, iter8, ns9, iter9)
end Test

end BasicIter
end Index
end TensorLib
