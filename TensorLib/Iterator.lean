/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/

import TensorLib.Common

/-!
Iterators play a big role in TensorLib.
-/
namespace TensorLib

/-
An iterator that can be reset to its starting point.
I spent a long time trying to use `next : iter -> Option (value × iter)` but separating
them came out much simpler.
-/
class Iterator (iter value : Type) where
  next : iter -> Option iter
  peek : iter -> value
  size : iter -> Nat
  reset : iter -> iter
namespace Iterator

-- https://leanprover-community.github.io/archive/stream/270676-lean4/topic/cannot.20find.20synthesization.20order.20for.20instance.html
set_option synthInstance.checkSynthOrder false

instance forInInstance [Monad m] [inst : Iterator iter value] : ForIn m iter value where
  forIn {α} [Monad m] (iter : iter) (x : α) (f : value -> α -> m (ForInStep α)) : m α := do
    let mut iter := iter
    let mut res := x
    for _ in [0:inst.size iter] do
      let n := inst.peek iter
      match <- f n res with
      | .yield k =>
        res := k
      | .done k =>
        res := k
        break
      match inst.next iter with
      | .none => break
      | .some iter' =>
        iter := iter'
    return res

def toList [Iterator iter value] (iter : iter) : List value := Id.run do
  let mut res := []
  for xs in iter do
    res := xs :: res
  return res.reverse

structure NatIter where
  private mk::
  private lo : Nat
  private hi : Nat
  private current : Nat

namespace NatIter

def makeWithLower (lo hi : Nat) : NatIter := mk lo hi lo

def make (hi : Nat) : NatIter := makeWithLower 0 hi

instance instNat : Iterator NatIter Nat where
  peek iter := iter.current
  next iter := if iter.hi <= iter.current + 1 then none else some {iter with current := iter.current + 1}
  size iter := iter.hi - iter.lo
  reset iter := { iter with current := iter.lo }

#guard instNat.size (make 10) == 10
#guard instNat.size (make 10) == 10
#guard instNat.toList (make 5) == [0, 1, 2, 3, 4]
#guard instNat.size (makeWithLower 3 10) == 7
#guard instNat.toList (makeWithLower 3 5) == [3, 4]

private def testBreak (iter : NatIter) : List Nat := Id.run do
  let mut res := []
  for xs in iter do
    res := xs :: res
    break
  return res.reverse

#guard testBreak (make 10) == [0]

private def testReturn (iter : NatIter) : List Nat := Id.run do
  let mut res := []
  let mut i := 0
  for xs in iter do
    res := xs :: res
    i := i + 1
    if i == 3 then return res.reverse
  return res.reverse

#guard testReturn (make 10) == [0, 1, 2]

end NatIter

structure IntIter where
  private mk ::
  private start : Int
  private stop : Int
  private step : Int
  private peek : Int
  private stepNz : step ≠ 0
deriving Repr

instance : Inhabited IntIter where
  default := IntIter.mk 0 1 1 0 (by simp)

namespace IntIter

def make (start stop step : Int) : Err IntIter :=
  if H : step == 0 then .error "step can't be 0" else
  let stepNz : step ≠ 0 := by simp_all
  if step < 0 && start < stop then .error "start below stop with negative step"
  else if 0 <= step && stop < start then .error "stop below start with positive step"
  else .ok $ IntIter.mk start stop step start stepNz

def make! (start stop step : Int) : IntIter := get! $ make start stop step

def reset (iter : IntIter) : IntIter := { iter with peek := iter.start }

def size (iter : IntIter) : Nat := natDivCeil (iter.stop - iter.start).natAbs iter.step.natAbs

def next (iter : IntIter) : Option IntIter :=
  let n := iter.peek + iter.step
  let iter' := { iter with peek := n }
  if iter.step < 0 then
    if n <= iter.stop then none else some iter'
  else
    if iter.stop <= n then none else some iter'

instance instInt : Iterator IntIter Int where
  reset := reset
  size := size
  peek := peek
  next := next

#guard instInt.size (make! 0 10 1) == 10
#guard instInt.size (make! 0 10 2) == 5
#guard instInt.size (make! 0 10 3) == 4
#guard instInt.size (make! 5 10 2) == 3
#guard !(make 0 (-1) 1).isOk
#guard instInt.size (make! (-5) (-10) (-1)) == 5
#guard !(make (-10) (-5) (-1)).isOk

#guard instInt.toList (make! 5 10 2) == [5, 7, 9]
#guard instInt.toList (make! (-5) (-10) (-2)) == [-5, -7, -9]
#guard instInt.toList (make! (-5) (-11) (-2)) == [-5, -7, -9]
#guard instInt.toList (make! (-5) (-12) (-2)) == [-5, -7, -9, -11]
#guard instInt.toList (make! (5) (-5) (-2)) == [5, 3, 1, -1, -3]

end IntIter

section Pairs

-- This isn't used, but is a nice example.
-- Assumes the size of the two iterators is the same
local instance instPairLockStep [inst1 : Iterator i1 v1] [inst2 : Iterator i2 v2] : Iterator (i1 × i2) (v1 × v2) where
  size := fun (l, r) => min (inst1.size l) (inst2.size r)
  reset := fun (l, r) => (inst1.reset l, inst2.reset r)
  peek := fun (l, r) => (inst1.peek l, inst2.peek r)
  next := fun (l, r) =>
    match inst1.next l, inst2.next r with
    | .some l, .some r => some (l, r)
    | _, _ => none

#guard
  let iter1 := NatIter.make 5
  let iter := (iter1, iter1)
  Iterator.toList iter == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]

-- A carry-adder-style pair iterator
instance instPairCarry [inst1 : Iterator i1 v1] [inst2 : Iterator i2 v2] : Iterator (i1 × i2) (v1 × v2) where
  size := fun (l, r) => inst1.size l * inst2.size r
  reset := fun (l, r) => (inst1.reset l, inst2.reset r)
  peek := fun (l, r) => (inst1.peek l, inst2.peek r)
  next := fun (l, r) =>
    match inst1.next l with
    | .some l => some (l, r)
    | .none =>
      let l := inst1.reset l
       match inst2.next r with
      | .none => .none
      | .some r => .some (l, r)

#guard
  let iter1 := NatIter.make 1
  let iter2 := NatIter.make 1
  let iter := (iter1, iter2)
  Iterator.toList iter == [(0, 0)]

#guard
  let iter1 := NatIter.make 5
  let iter2 := NatIter.make 3
  let iter := (iter1, iter2)
  Iterator.toList iter ==
    [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
     (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
     (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]

end Pairs

/-
Lists cause a problem for the `ForIn` instance. For example, when Lean sees

    for n in [i1, i2, i3] do
      ...

if we have an Iterator instance for List, it may either use the usual List `ForIn`, or
the one generated by the Iterator instance. When I tried replacing all the old manual iterators
with the class-generated ones, I got many failures to find a `ForIn` instance in the alloted cycles.
There are ways to try to hack the precedence, but let's just box the lists of iterators.
-/
section Lists

-- This iterates in "little-endian" order; the left-most element increases fastest
structure LEList (iter : Type) where
  private mk::
  val : List iter

-- We advance both big-endian and little-endian iterators the same way
-- TODO: Use a zipper or some other data structure so we don't need to
-- rebuild the list from scratch on each advance.
private def advance [inst : Iterator iter value] (iters : List iter) (acc : List iter) : Option (List iter) :=
  match iters with
  | [] => none
  | iter :: iters =>
    match inst.next iter with
    | .some iter => acc.reverse ++ iter :: iters
    | .none => @advance _ _ inst iters (inst.reset iter :: acc)

namespace LEList

def make (iters : List iter) : LEList iter := LEList.mk iters

-- This iterates in "little-endian" order; the left-most element increases fastest
instance instListLE [inst : Iterator iter value] : Iterator (LEList iter) (List value) where
  size iter := iter.val.foldl (fun acc i => acc * inst.size i) 1
  reset iter := LEList.mk $ iter.val.map inst.reset
  peek iter := iter.val.map inst.peek
  next iter := (@advance _ _ inst iter.val []).map LEList.mk

#guard
  let iter1 := NatIter.make 5
  let iter := LEList.mk [iter1]
  instListLE.toList iter == [[0], [1], [2], [3], [4]]

#guard
  let iter1 := NatIter.make 5
  let iter2 := NatIter.make 3
  let iter := LEList.mk [iter1, iter2]
  instListLE.toList iter ==
    [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0],
     [0, 1], [1, 1], [2, 1], [3, 1], [4, 1],
     [0, 2], [1, 2], [2, 2], [3, 2], [4, 2]]

end LEList

-- This iterates in "big-endian" order; the right-most element increases fastest
structure BEList (iter : Type) where
  private mk::
  val : List iter

namespace BEList

def make (iters : List iter) : BEList iter := BEList.mk iters.reverse

instance instListBE [inst : Iterator iter value] : Iterator (BEList iter) (List value) where
  size iter := iter.val.foldl (fun acc i => acc * inst.size i) 1
  reset iter := BEList.mk $ iter.val.map inst.reset
  peek iter := (iter.val.map inst.peek).reverse
  next iter := (@advance _ _ inst iter.val []).map BEList.mk

#guard
  let iter1 := NatIter.make 5
  let iter := BEList.make [iter1]
  instListBE.toList iter == [[0], [1], [2], [3], [4]]


#guard
  let iter1 := NatIter.make 5
  let iter2 := NatIter.make 3
  let iter := BEList.make [iter1, iter2]
  instListBE.toList iter ==
    [[0, 0], [0, 1], [0, 2],
     [1, 0], [1, 1], [1, 2],
     [2, 0], [2, 1], [2, 2],
     [3, 0], [3, 1], [3, 2],
     [4, 0], [4, 1], [4, 2]]

end BEList
end Lists


end Iterator
end TensorLib
