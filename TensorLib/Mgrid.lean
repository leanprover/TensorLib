/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/

import TensorLib.Common
import TensorLib.Tensor
import TensorLib.Slice
import TensorLib.Index

namespace TensorLib
namespace Mgrid

--! All mgrid array elements are BV64
abbrev elementType : Type := BV64

/-
`mgrid` dimensions, being infinite in both directions, are sized somewhat differently
from slices applied to normal arrays. Both start and stop must be nonempty.

Because of the unintuitive way NumPy sets defaults for start/stop in `mgrid` slices, we
require both are non-empty here. For example,

# np.mgrid[-5::-1]
array([ 0, -1, -2, -3, -4])

Here NumPy infers the stop to be 0.

# np.mgrid[-5::1]
array([], dtype=int64)

Here you may expect the same, and to get [-5, -4, -3, -2, -1], but instead
we get an empty array.
-/

/-!
Iterating over an `mgrid`, which is essentially an infinite array in both directions in any dimension,
is fundmentally different from iterating over a fixed size array. The iteration code is very similar, but
combining them made the code overly complex. For example, in the finite case we have Nat indices, and here
we have Int indices. Details like this made the combination hard to manage.
-/
structure Iter where
  private mk ::
  private start : Int
  private stop : Int
  private step : Int
  private curr : Option Int -- not returned yet, `curr = none` iff the iterator is empty
  private stepNz : step ≠ 0
deriving Repr

instance : Inhabited Iter where
  default := Iter.mk 0 0 1 none H
    where H : 1 ≠ 0 := by omega

namespace Iter

def peek (iter : Iter) : Option Int := iter.curr

def reset (iter : Iter) : Iter := { iter with curr := some iter.start }

def sliceOf (iter : Iter) : Slice :=
  let H : some iter.step ≠ some 0 := by simp_all [iter.stepNz]
  Slice.mk (some iter.start) (some iter.stop) (some iter.step) H

def make (slice : Slice) : Option Iter :=
  let step := slice.step.getD 1
  let H : step ≠ 0 := by
    have H1 := slice.stepNz
    unfold step
    cases H : slice.step <;> simp_all
  match slice.start, slice.stop with
  | some start, some stop => some $ Iter.mk start stop step (some start) H
  | _, _ => none

private def make! (slice : Slice) : Iter := match make slice with
| none => panic "illegal slice"
| some iter => iter

private def dir (iter : Iter) : Slice.Dir := if iter.step < 0 then .Backward else .Forward

def size (iter : Iter) : Nat :=
  let sz : Int := match iter.dir with
  | .Forward => (iter.stop - iter.start) / iter.step
  | .Backward => (iter.start - iter.stop) / (-iter.step)
  sz.toNat

def next (iter : Iter) : Option (Int × Iter) := match iter.curr with
| .none => .none
| .some curr =>
  match iter.dir with
  | .Forward =>
    let c := curr + iter.step
    let nextCurr := if iter.stop <= c then none else some c
    some (curr, { iter with curr := nextCurr })
  | .Backward =>
    let c := curr + iter.step
    let nextCurr := if c <= iter.stop then none else some c
    some (curr, { iter with curr := nextCurr })

def hasNext (iter : Iter) : Bool := iter.next.isSome

instance [Monad m] : ForIn m Iter Int where
  forIn {α} [Monad m] (iter : Iter) (x : α) (f : Int -> α -> m (ForInStep α)) : m α := do
    let mut iter := iter
    let mut res := x
    for _ in [0:iter.size] do
      match iter.next with
      | .none => break
      | .some (n, iter') =>
        iter := iter'
        match <- f n res with
        | .yield k =>
          res := k
        | .done k =>
          res := k
          break
    return res

private def toList (iter : Iter) : List Int := Id.run do
  let mut res := []
  for xs in iter do
    res := xs :: res
  return res.reverse

private partial def toList' (iter : Iter) (acc : List Int := []) : List Int :=
  match iter.next with
  | none => acc.reverse
  | some (n, iter) => toList' iter (n :: acc)

#guard (Iter.make! (Slice.make! (.some 3) (.some 6) .none)).toList == [3, 4, 5]
#guard (Iter.make! (Slice.make! (.some 3) (.some 6) .none)).toList' == [3, 4, 5]
#guard (Iter.make! (Slice.make! (.some 6) (.some 3) (.some (-1)))).toList == [6, 5, 4]
#guard (Iter.make! (Slice.make! (.some 6) (.some 3) (.some (-1)))).toList' == [6, 5, 4]

end Iter

def size (slice : Slice) : Option Nat := (Iter.make slice).map Iter.size

structure MgridIter where
  private mk ::
  private iters : List Iter
  private done : Bool
deriving Inhabited, Repr

namespace MgridIter

def ndim (iter : MgridIter) : Nat := iter.iters.length

def make (iters : List Iter) : MgridIter := MgridIter.mk iters.reverse false

def componentSize (iter : MgridIter) : Nat := iter.iters.foldl (fun acc iter => acc * iter.size) 1

def size (iter : MgridIter) : Nat := iter.componentSize * iter.ndim

-- The (reversed) current index iteration (not yet returned)
-- The peek is .some by invariant
private def current (iters : List Iter) : List Int := iters.map (fun iter => iter.peek.getD impossible)

-- Returns the current element (that hasn't yet been returned), and the next iterator.
private def nextWithCarry (iters : List Iter) (carry : Bool) : List Int × Option (List Iter) :=
  match iters with
  | [] => ([], if carry then none else some [])
  | iter :: iters =>
    match iter.next with
    | .none => impossible -- We already reset the iterator once we return the max element
    | .some (n, nextIter) =>
      if nextIter.hasNext
      then (n :: current iters, nextIter :: iters)
      else
        let iter := iter.reset
        let (ns, iters) := nextWithCarry iters (carry := true)
        let iters := iters.map fun b => iter :: b
        (n :: ns, iters)

def next (giter : MgridIter) : Option (List Int × MgridIter) :=
  if giter.done then none else
  let (ns, iters) := nextWithCarry giter.iters false
  let ns := ns.reverse
  let giter : MgridIter := match iters with
  | none => { giter with done := true } -- All slice iterators are maxxed out in this case. Should we check this?
  | some iters => { giter with iters }
  some (ns, giter)

instance [Monad m] : ForIn m MgridIter (List Int) where
  forIn {α} [Monad m] (iter : MgridIter) (x : α) (f : List Int -> α -> m (ForInStep α)) : m α := do
    let mut iter : MgridIter := iter
    let mut res := x
    for _ in [0:iter.componentSize] do
      match iter.next with
      | .none => break
      | .some (ns, iter') =>
        iter := iter'
        match <- f ns res with
        | .yield k => res := k
        | .done k => return k
    return res

private def toList (iter : MgridIter) : List (List Int) := Id.run do
  let mut res := []
  for xs in iter do
    res := xs :: res
  return res.reverse

#guard
  let iter := Iter.make! (Slice.ofStop 3)
  (MgridIter.make [iter, iter]).toList ==
    [[0, 0], [0, 1], [0, 2],
     [1, 0], [1, 1], [1, 2],
     [2, 0], [2, 1], [2, 2]]

#guard
  let iter0 := Iter.make! (Slice.ofStop 2)
  let iter1 := Iter.make! (Slice.make! (some 4) (some 0) (some (-1)))
  (MgridIter.make [iter0, iter1]).toList ==
  [[0, 4], [0, 3], [0, 2], [0, 1],
   [1, 4], [1, 3], [1, 2], [1, 1]]

#guard
  let iter0 := Iter.make! $ Slice.ofStartStop (-4) (-2)
  let iter1 := Iter.make! $ Slice.make! (some 7) (some 4) (some (-1))
  let giter := MgridIter.make [iter0, iter1]
  giter.toList == [
    [-4, 7], [-4, 6], [-4, 5],
    [-3, 7], [-3, 6], [-3, 5]
  ]

end MgridIter

private def sliceSize (slice : Slice) : Option Nat := (Iter.make slice).map Iter.size

#guard (sliceSize Slice.all).isNone
#guard (sliceSize (Slice.make! .none (.some 5) .none)).isNone
#guard (sliceSize (Slice.make! (.some 5) .none .none)).isNone
#guard (sliceSize (Slice.make! (.some 5) (.some 10) .none)) == .some 5
#guard (sliceSize (Slice.make! (.some (-5)) (.some (-10)) .none)) == .some 0
#guard (sliceSize (Slice.make! (.some (-5)) (.some (-10)) (.some (-1)))) == .some 5
#guard (sliceSize (Slice.make! (.some (-5)) (.some 10) (.some (-1)))) == .some 0
#guard (sliceSize (Slice.make! (.some (-5)) (.some 10) (.some 1))) == .some 15
#guard (sliceSize (Slice.make! (.some (-5)) (.some 10) (.some (-3)))) == .some 0
#guard (sliceSize (Slice.make! (.some (-5)) (.some 10) (.some 3))) == .some 5

end Mgrid

/-
`mgrid` is unique in that it can be considered as a Nat array (in both postivie and negative directions)
of arbitraray dimensions. In a slice `[start:stop:step]` applied to `mgrid`, `start = -1`
doesn't correspond to an easy way to refer to the last element; it just means the number `-1`.

Because we index with slices into `mgrid` differently, we use a new iterator type to avoid
confusion.

The shape of the array resulting from `mgrid[slices]` is |slices| :: slices.map(fun x => x.shape).

Slices are treated differently in NumPy when used in mgrid vs indexing. In particular,
since we don't know what the stopping point should be from the context, it can use the `start`
value as a `stop` instead.  For example,

  # np.arange(10)[5:]
  [5, 6, 7, 8, 9]

  # np.mgrid[5:]
  [0, 1, 2, 3, 4]

However, things work normally when there is a stopping point

  # np.mgrid[5:10]
  [5, 6, 7, 8, 9]

Since this is imo surprising, we just fail if the start or stopping point are absent.
All mgrid ints are stored as 64-bit, little-endian ints by convention. In Numpy they are stored with
native byte order, as the architecture word size.

For example,

# np.mgrid[2:4:, 4:7:]
array([[[2, 2, 2],
        [3, 3, 3]],

       [[4, 5, 6],
        [4, 5, 6]]])

We have an iterator that will give us the indices [0, 0], [0, 1], [0, 2], [1, 0], ..., [1, 2]
and an iterator that will give us the values [2, 4], [2, 5], [2, 6], ... , [3, 6]
These are the same length, so we can combine them to get the mgrid;
- [0, 0] of the first element gets 2,
- [0, 0] of the second element gets 4
- [0, 1] of the first gets 2
- [0, 1] of the second gets 5
- ...

While experimenting, I noticed the following wart in np.mgrid

    # np.mgrid[1:1:1, 0:10]
    array([], shape=(2, 0, 10), dtype=int64)

    # np.mgrid[1:1:1]
    array([], dtype=int64)

    # np.mgrid[2:1:1]
    array([], dtype=int64)

    # np.mgrid[1:1:1, 0:10]
    array([], shape=(2, 0, 10), dtype=int64)

    # np.mgrid[2:1:1, 0:10]
    ValueError: negative dimensions are not allowed

There's something weird with backward empty slices.
I think the last one should be the same as the others, just a size-0 dimension.
-/
def mgrid (slices : List Slice) : Err Tensor := do
  let sliceCount := slices.length
  let mut slicesDims : List Nat := []
  for slice in slices.reverse do
    match Mgrid.sliceSize slice with
    | none => .error "Illegal slice"
    | some sz => slicesDims := sz :: slicesDims
  let shape := Shape.mk $ sliceCount :: slicesDims
  let slicesShape := Shape.mk slicesDims
  let dtype := Dtype.int64 -- We fix a little-endian uint64 by convention
  let mut arr := Tensor.zeros dtype shape
  match slices.mapM fun s => Mgrid.Iter.make s with
  | none => .error "Illegal slice" -- redundant with the sliceSize check above
  | some iters =>
  let mut mgridIter := Mgrid.MgridIter.make iters
  let indexIter := DimsIter.make slicesShape
  if mgridIter.componentSize != indexIter.size then .error "Invariant failure: iterator size mismatch at start"
  for index in indexIter do
    match mgridIter.next with
    | .none => .error "Invariant failure: iterator size mismatch during iteration"
    | .some (values, mgridIter') =>
      mgridIter := mgridIter'
      if values.length != sliceCount then .error "Invariant failure: value length mismatch"
      for (i, v) in (List.range sliceCount).zip values do
        let value := Int64.ofInt v
        arr <- Tensor.Element.setDimIndex arr (i :: index) value
  return arr

section Test

open Tensor.Format

abbrev tp := Mgrid.elementType
private def mg (slices : List Slice) : Tree tp := get! $ (get! (mgrid slices)).toTree tp

#guard (mg [Slice.ofStartStop 2 4, Slice.ofStartStop 4 7]) ==
  .node [
    .node [
      .root [2, 2, 2], .root [3, 3, 3]
    ],
    .node [
      .root [4, 5, 6], .root [4, 5, 6]
    ]
 ]

-- 0D
#guard mg [] == .root []

-- 1D
#guard mg [Slice.ofStartStop (-4) (-2)] == .node [.root [-4, -3]]
#guard mg [Slice.ofStartStop (-4) 2] == .node [.root [-4, -3, -2, -1, 0, 1]]
#guard mg [Slice.make! (some 2) (some (-4)) (some (-1))] == .node [.root [2, 1, 0, -1, -2, -3]]

-- 2D

#guard mg [Slice.ofStartStop (-4) (-2), Slice.make! (some 7) (some 4) (some (-1))] ==
  .node [
    .node [
      .root [-4, -4, -4], .root [-3, -3, -3] -- .root [-3, -3, -3]
    ],
    .node [
      .root [7, 6, 5], .root [7, 6, 5] -- [7, 6, 5]
    ]
  ]

-- 3D

#guard (get! (mgrid [Slice.ofStartStop 2 4, Slice.ofStartStop 4 7, Slice.ofStop 2])).toTree BV64 == .ok (
  .node [
    .node [
      .node [
        .root [2, 2], .root [2, 2], .root [2, 2]
      ],
      .node [
        .root [3, 3], .root [3, 3], .root [3, 3]
      ],
    ],
    .node [
      .node [
        .root [4, 4], .root [5, 5], .root [6, 6]
      ],
      .node [
        .root [4, 4], .root [5, 5], .root [6, 6]
      ]
    ],
    .node [
      .node [
        .root [0, 1], .root [0, 1], .root [0, 1]
      ],
      .node [
        .root [0, 1], .root [0, 1], .root [0, 1]
      ]
    ]
  ]
)

end Test
end TensorLib
