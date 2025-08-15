/-
Copyright TensorLib Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-/

import Aesop
import TensorLib.Common
import TensorLib.Iterator

namespace TensorLib

/-!
Slices are triples of start/stop/step, all of which can be positive, negative, or missing.
The only restriction is that `step` can not be 0.

Slices coming from NumPy allows negative numbers. These are interpeted
as offsets from the end of the array (start/stop) and as going backward rather
than forward (step).

Here are some behaviors of slicing I found surprising:

1. No out-of-bounds

Int indices cause errors when out of bounds, but slices are never out of bounds
# x = np.arange(5)[]
# x[5] is an error, out of bounds
# x[5:] = x[5::] = x[500::] = []

Any values are legal here.

# x[5:100:-100] = []

2. Negative steps can't (easily) access the first element with an int `stop`
`start` is inclusive, while `stop` is exclusive. This works nicely when `step` is
positive, but not so much when it is negative.

# x[5:0:-1] = [4, 3, 2, 1]

If n = -1 .. -9, then x[n] == x[10 + n] which always gives some strict subset of the
values

# x[5:-2:-1] = x[5:3:-1] = [4]
# x[5:-5:-1] = x[5:0:-1] = [4, 3, 2, 1]

You need to use -n-1 (or smaller) to reach it

# x[5:-6:-1] = x[5:-100:-1] = [4, 3, 2, 1, 0]

There is thus a surprising discontinuity between `stop` = 0 and `stop` -1

# x[5:0:-1] = [4, 3, 2, 1]
# x[5:-1:-1] = []
# x[5:-2:-1] = [4]
...
# x[5:-5:-1] = [4, 3, 2, 1]
# x[5:-6:-1] = [4, 3, 2, 1, 0]

# x
array([0, 1, 2, 3, 4])

# [x[i:] for i in range(-10, 10, 1)]
[array([0, 1, 2, 3, 4]),
 array([0, 1, 2, 3, 4]),
 array([0, 1, 2, 3, 4]),
 array([0, 1, 2, 3, 4]),
 array([0, 1, 2, 3, 4]),
 array([0, 1, 2, 3, 4]),
 array([1, 2, 3, 4]),
 array([2, 3, 4]),
 array([3, 4]),
 array([4]),
 array([0, 1, 2, 3, 4]),
 array([1, 2, 3, 4]),
 array([2, 3, 4]),
 array([3, 4]),
 array([4]),
 array([], dtype=int64),
 array([], dtype=int64),
 array([], dtype=int64),
 array([], dtype=int64),
 array([], dtype=int64)]

If we know the size of the array/dimension we are slicing, this can
all be substantially simplified. With a known dimension, we will compile
the slice semantics to a more clear datatype.

-/
namespace Slice

inductive Dir where
| Forward
| Backward

end Slice

/-
Defaults for missing parts of the triple are well defined by the docs:
https://numpy.org/doc/stable/user/basics.indexing.html#slicing-and-striding

> Defaults for i:j:k
>
> Assume n is the number of elements in the dimension being sliced.
> Then, if i is not given it defaults to 0 for k > 0 and n - 1 for k < 0 .
> If j is not given it defaults to n for k > 0 and -n-1 for k < 0 .
> If k is not given it defaults to 1.

Negative values are allowed. The meaning of x[-k] is x[n-k] where `n` is the size of the
relevant dimension.
Note that these requirements from the NumPy docs don't make sense if `n = 0`.
-/
structure Slice where
  start : Option Int
  stop : Option Int
  step : Option Int
  stepNz : step ≠ .some 0
deriving BEq, Repr

instance : Inhabited Slice where
  default := Slice.mk .none .none .none (by simp)

namespace Slice

def make (start stop step : Option Int) : Err Slice :=
  match H : step with
  | .none =>
    let stepNz : step ≠ some 0 := by rw [H]; trivial
    .ok (Slice.mk start stop step stepNz)
  | .some k =>
    if H1 : k == 0 then .error "step can't be 0" else
    let stepNz : step ≠ some 0 := by rw [H]; simp_all
    .ok (Slice.mk start stop step stepNz)

partial def make! (start stop step : Option Int) : Slice :=
  get! (make start stop step)

def all : Slice := default

def ofInt (n : Int) : Slice :=
  let stepNz : Option.none ≠ .some (0:Int) := by trivial
  Slice.mk (.some n) .none .none stepNz

def ofStop (n : Int) : Slice :=
  let stepNz : Option.none ≠ .some (0:Int) := by trivial
  Slice.mk (.some 0) (.some n) .none stepNz

def ofStartStop (start stop : Int) : Slice :=
  let stepNz : Option.none ≠ .some (0:Int) := by trivial
  Slice.mk (.some start) (.some stop) .none stepNz

def dir (s : Slice): Dir := match s.step with
  | .none => Dir.Forward
  | .some k => if k < 0 then Dir.Backward else Dir.Forward

/-
Calculating the nat start index of a slice is complicated by the
NumPy defaults that differ based on direction (forward/backward)
and negative values.
-/
def startOrDefault (s : Slice) (dim : Nat) : Nat :=
  let dir := s.dir
  match s.start with
  | .none => match dir with
    | Dir.Forward => 0
    | Dir.Backward => dim - 1 -- if dim = 0 then start = 0, which seems the reasonable choice
  | .some k =>
    if k < -dim then 0
    else if -dim <= k && k < 0 then (dim + k).toNat
    else if 0 <= k && k < dim then k.toNat
    else /- n <= k -/ match dir with
    | Dir.Forward => dim
    | Dir.Backward => dim - 1

/-
We use an option here to signal that the stopping point for negative step
is the first element of the list. We can't use 0 because the stopping point
is exclusive in a slice. [start ... stop) We can't use any negative number
because the first `n` negative numbers wrap back around to positive numbers.

I would love to find a representation with straightfoward arithmetic semantics
that does not require an option for stop, but I could not find one in the presence of
both postive and negative steps in about an hour of brainstorming.
-/
def stopOrDefault (s : Slice) (dim : Nat) : Option Nat :=
  match s.stop, s.dir with
  | .none, .Forward => .some dim
  | .none, .Backward => .none
  | .some k, .Forward =>
    if k < -dim then .some 0
    else if -dim <= k && k < 0 then .some (dim + k).toNat
    else if 0 <= k && k < dim then .some k.toNat
    else /- n <= k -/ .some dim
  | .some k, .Backward =>
    if k < -dim then .none
    else if -dim <= k && k < 0 then .some (dim + k).toNat
    else if 0 <= k && k < dim then .some k.toNat
    else /- n <= k -/ .some dim

theorem stopOrDefaultForward (s : Slice) (dim : Nat) :
  match s.dir with
  | .Forward => (s.stopOrDefault dim).isSome
  | .Backward => True := by
  unfold stopOrDefault
  aesop

theorem stopRange (s : Slice) (dim : Nat) :
  match s.stopOrDefault dim with
  | .none => True
  | .some k => k <= dim
:= by
  unfold stopOrDefault
  cases s.stop
  . cases s.dir
    . simp
    . simp
  . rename_i k
    cases s.dir
    . simp
      by_cases H : k < -dim
      all_goals simp [H]
      by_cases H1 : k < 0
      . aesop (config := { warnOnNonterminal := false })
        all_goals omega
      . aesop (config := { warnOnNonterminal := false })
        all_goals omega
    . simp
      by_cases H : k < -dim
      all_goals simp [H]
      by_cases H1 : k < 0
      . simp [H1]
        aesop (config := { warnOnNonterminal := false })
        all_goals omega
      . aesop (config := { warnOnNonterminal := false })
        all_goals omega

theorem stopForward (s : Slice) (dim : Nat) (H : s.dir = .Forward) : (s.stopOrDefault dim).isSome := by
  unfold stopOrDefault
  aesop

def stepOrDefault (s : Slice) : Int := s.step.getD 1

theorem stepOrDefaultNz (s : Slice) : ¬ s.stepOrDefault = 0 := by
  unfold stepOrDefault
  cases H : s.step
  . simp
  . let H1 := s.stepNz
    aesop

theorem stepForward {s : Slice} (H : s.dir = .Forward) : 0 < s.stepOrDefault := by
  revert H
  unfold stepOrDefault dir
  cases H1 : s.step
  all_goals simp
  rename_i k
  have H2 := s.stepNz
  aesop (config := { warnOnNonterminal := false })
  omega

def defaults (s : Slice) (dim : Nat) : Nat × Option Nat × Int :=
  (s.startOrDefault dim, s.stopOrDefault dim, s.stepOrDefault)

def size (s : Slice) (dim : Nat) : Nat :=
  -- Without a special case for dim = 0, I couldn't figure out how to handle the case when the step is (-1)
  if dim == 0 then 0 else
  let (start, stop, step) := s.defaults dim
  match s.dir, stop with
  | .Forward, .none => 0 -- Actually impossible due to `stopForward`, but 0 is a sensible default
  | .Forward, .some stop => natDivCeil (stop - start) step.toNat
  | .Backward, .none => (start + 1) / step.natAbs -- This is the case that fails when dim = 0 and step = -1
  | .Backward, .some stop => (start - stop) / step.natAbs

#guard (make! .none .none .none).size 0 == 0
#guard (make! (.some 0) (.some 10) (.some 1)).size 0 == 0
#guard (make! (.some 10) (.some 0) (.some (-1))).size 0 == 0
#guard (make! .none .none (.some (-1))).size 0 == 0
#guard (make! (.some 0) .none (.some (-1))).size 0 == 0
#guard (make! (.some (-5)) .none (.some (-1))).size 0 == 0
#guard (make! .none .none .none).size 10 == 10
#guard (make! .none .none (.some (-1))).size 10 == 10
#guard (make! .none .none (.some (-2))).size 10 == 5
#guard (make! .none .none (.some 2)).size 10 == 5
#guard (make! .none .none (.some 2)).size 5 == 3
#guard (make! (.some 5) .none .none).size 10 == 5
#guard (make! (.some 5) .none (.some 1)).size 10 == 5
#guard (make! (.some 5) .none (.some 2)).size 10 == 3
#guard (make! (.some 5) .none (.some 3)).size 10 == 2
#guard (make! (.some 5) .none (.some (-1))).size 10 == 6
#guard (make! (.some 5) .none (.some (-3))).size 10 == 2
#guard (make! .none (.some 5) .none).size 10 == 5
#guard (make! .none (.some 5) (.some (-1))).size 10 == 4
#guard (make! (.some 100) .none (.some (-1))).size 10 == 10 -- ok if start is past the end of the array
#guard (make! (.some 100) (.some (-2)) (.some (-1))).size 10 == 1

private theorem size0 (s : Slice) : s.size 0 = 0 := by
  unfold size defaults stopOrDefault natDivCeil startOrDefault
  cases H0 : s.dir <;> cases H1 : s.start <;> cases H2 : s.stop <;> simp

private theorem add_div_le {a b c d : Nat} (H1 : b < c) (H2 : a <= d) : (a + b) / c ≤ d := by
  rw [Nat.div_le_iff_le_mul] <;> try omega
  cases d <;> try omega
  rename_i d
  suffices H3 : a + b ≤ (d+1) * c + (c - 1) by omega
  apply (@Nat.add_le_add a ((d+1)*c) b (c-1)) <;> try omega
  apply (@Nat.le_trans a (d+1)) <;> try omega
  apply (@Nat.le_mul_of_pos_right c (d+1))
  omega

private theorem add_sub_assoc (n m k : Nat) (H : k <= m) : n + m - k = n + (m - k) := by omega

private theorem le_div_le (c : Nat) (H : a ≤ b) : a / c ≤ b := by
  cases c ; simp
  rename_i c
  rw [Nat.div_le_iff_le_mul, Nat.mul_add] <;> try omega

private theorem div_succ (k s : Nat) : (k + s) / s ≤ k + 1 := by
  cases s <;> try omega
  simp
  rename_i n
  apply le_div_le
  omega

private theorem sizeDim (s : Slice) (dim : Nat) : s.size dim <= dim := by
  cases H_dim : dim ; simp [size0]
  rename_i k
  unfold size defaults stopOrDefault natDivCeil startOrDefault
  simp
  cases H0 : s.dir
  . have H8 := stepForward H0
    cases H1 : s.start <;> cases H2 : s.stop <;> generalize H4 : s.stepOrDefault.toNat = step <;> simp_all
    . apply div_succ
    . rename_i a
      by_cases H5 : a < -(k+1) <;> simp [H5]
      . rw [Nat.div_eq_of_lt] <;> omega
      . by_cases H6 : a < 0 <;> simp [H6]
        . simp [show -(↑k + 1) ≤ a by omega]
          rw [add_sub_assoc] <;> try omega
          apply add_div_le <;> omega
        . simp [show 0 <= a by omega]
          by_cases H7 : a < ↑k + 1 <;> simp [H7]
          . rw [add_sub_assoc] <;> try omega
            apply add_div_le <;> omega
          . apply div_succ
    . rename_i a
      by_cases H5 : a < -(k+1) <;> simp [H5]
      . apply div_succ
      . by_cases H6 : a < 0 <;> simp [H6]
        . simp [show -(↑k + 1) ≤ a by omega]
          rw [add_sub_assoc] <;> try omega
          apply add_div_le <;> omega
        . simp [show 0 <= a by omega]
          by_cases H7 : a < ↑k + 1 <;> simp [H7]
          . rw [add_sub_assoc] <;> try omega
            apply add_div_le <;> omega
          . rw [Nat.div_eq_of_lt] <;> omega
    . rename_i b a
      by_cases H5 : a < -(k+1) <;> simp [H5]
      . rw [Nat.div_eq_of_lt] <;> omega
      . simp [show -(↑k + 1) ≤ a by omega]
        . by_cases H6 : a < 0 <;> simp [H6]
          . by_cases H7 : b < -(k+1) <;> simp [H7]
            . rw [add_sub_assoc] <;> try omega
              apply add_div_le <;> omega
            . simp [show -(↑k + 1) ≤ b by omega]
              by_cases H9 : b < 0 <;> simp [H9]
              . rw [add_sub_assoc] <;> try omega
                apply add_div_le <;> omega
              . simp [show 0 ≤ b ∧ -(↑k + 1) ≤ b by omega]
                . by_cases H10 : b < k+1 <;> simp [H10]
                  . rw [add_sub_assoc] <;> try omega
                    apply add_div_le <;> omega
                  . rw [add_sub_assoc] <;> try omega
                    apply add_div_le <;> omega
          . simp [show 0 <= a by omega]
            by_cases H7 : a < ↑k + 1 <;> simp [H7]
            . rw [add_sub_assoc] <;> try omega
              apply add_div_le <;> omega
            . by_cases H9 : b < 0
              . rw [add_sub_assoc] <;> try omega
                apply add_div_le <;> omega
              . simp [show 0 ≤ b ∧ -(↑k + 1) ≤ b by omega]
                . by_cases H10 : b < k+1 <;> simp [H10]
                  . rw [add_sub_assoc] <;> try omega
                    apply add_div_le <;> omega
                  . rw [add_sub_assoc] <;> try omega
                    apply add_div_le <;> omega
  . cases H1 : s.start <;> cases H2 : s.stop <;> generalize H4 : s.stepOrDefault.toNat = step <;> simp_all
    . apply le_div_le ; simp
    . rename_i a
      by_cases H5 : a < -(k+1) <;> simp [H5]
      . apply le_div_le ; simp
      . by_cases H6 : a < 0 <;> simp [H6]
        . simp [show -(↑k + 1) ≤ a by omega]
          apply le_div_le
          omega
        . simp [show 0 <= a by omega]
          by_cases H7 : a < ↑k + 1 <;> simp [H7]
          . apply le_div_le ; omega
    . rename_i a
      by_cases H5 : a < -(k+1) <;> simp [H5]
      . apply le_div_le ; omega
      . by_cases H6 : a < 0 <;> simp [H6]
        . simp [show -(↑k + 1) ≤ a by omega]
          apply le_div_le ; omega
        . simp [show 0 <= a by omega]
          by_cases H7 : a < ↑k + 1 <;> simp [H7]
          . apply le_div_le ; omega
          . apply le_div_le ; omega
    . rename_i b a
      by_cases H5 : a < -(k+1) <;> simp [H5] <;> by_cases H6 : b < -(k+1) <;> simp [H6]
      . apply le_div_le ; omega
      . simp [show -(↑k + 1) ≤ b by omega]
        . by_cases H7 : b < 0 <;> simp [H7]
          . apply le_div_le ; omega
          . simp [show 0 <= b by omega]
            by_cases H8 : b < k+1 <;> simp [H8]
            . apply le_div_le ; omega
            . apply le_div_le ; omega
      . simp [show -(↑k + 1) ≤ a by omega]
        by_cases H7 : a < 0 <;> simp [H7]
        . simp [show 0 ≤ a by omega]
          by_cases H8 : a < k+1 <;> simp [H8]
      . simp [show -(↑k + 1) ≤ a by omega]
        by_cases H7 : a < 0 <;> simp [H7]
        . simp [show -(k+1) ≤ b by omega]
          by_cases H8 : b < 0 <;> simp [H8]
          . apply le_div_le ; omega
          . simp [show 0 ≤ b by omega]
            by_cases H9 : b < k+1 <;> simp [H9]
            . apply le_div_le ; omega
            . apply le_div_le ; omega
        . simp [show 0 ≤ a by omega]
          by_cases H10 : a < k+1 <;> simp [H10]
          . simp [show -(k+1) ≤ b by omega]
            by_cases H11 : b < 0 <;> simp [H11]
            . apply le_div_le ; omega
            . simp [show 0 ≤ b by omega]
              by_cases H12 : b < k+1 <;> simp [H12]
              . apply le_div_le ; omega
              . apply le_div_le ; omega
          . simp [show -(k+1) ≤ b by omega]
            by_cases H11 : b < 0 <;> simp [H11]
            . apply le_div_le ; omega
            . simp [show 0 ≤ b by omega]
              by_cases H12 : b < k+1 <;> simp [H12]
              . apply le_div_le ; omega

-- Reference implementation. When we do it for real we will use an iterator.
private partial def sliceList! [Inhabited a] (s : Slice) (xs : List a) : List a :=
  let n := xs.length
  if n == 0 then [] else
  -- 0 ≤ start ≤ n
  let (start, stop, step) := s.defaults n
  let done (i : Int) : Bool := i < 0 || n ≤ i || stop.any (fun k => k == i)
  -- TODO: prove termination
  let rec loop (acc : List a) (i : Int) : List a :=
    if done i then acc.reverse else
    -- TODO: prove indexing is in bounds
    loop (xs[i.toNat]! :: acc) (i + step)
  loop [] start

#guard (make! .none .none .none).sliceList! [0, 1, 2, 3, 4] == [0, 1, 2, 3, 4]
#guard (make! (.some 0) .none .none).sliceList! [0, 1, 2, 3, 4] == [0, 1, 2, 3, 4]
#guard (make! (.some 0) (.some 5) .none).sliceList! [0, 1, 2, 3, 4] == [0, 1, 2, 3, 4]
#guard (make! (.some 0) (.some 5) (.some 1)).sliceList! [0, 1, 2, 3, 4] == [0, 1, 2, 3, 4]
#guard (make! (.some 3) (.some 5) (.some 1)).sliceList! [0, 1, 2, 3, 4] == [3, 4]
#guard (make! (.some 10) (.some 5) (.some 1)).sliceList! [0, 1, 2, 3, 4] == []
#guard (make! (.some (-10)) (.some 5) (.some 1)).sliceList! [0, 1, 2, 3, 4] == [0, 1, 2, 3, 4]
#guard (make! (.some (-10)) (.some 5) (.some 100)).sliceList! [0, 1, 2, 3, 4] == [0]
#guard (make! .none .none (.some (-1))).sliceList! [0, 1, 2, 3, 4] == [4, 3, 2, 1, 0]
#guard (make! .none (.some 0) (.some (-1))).sliceList! [0, 1, 2, 3, 4] == [4, 3, 2, 1]
#guard (make! .none (.some (-5)) (.some (-1))).sliceList! [0, 1, 2, 3, 4] == [4, 3, 2, 1]
#guard (make! .none (.some (-6)) (.some (-1))).sliceList! [0, 1, 2, 3, 4] == [4, 3, 2, 1, 0]
#guard (make! .none (.some (-600)) (.some (-1))).sliceList! [0, 1, 2, 3, 4] == [4, 3, 2, 1, 0]
#guard (make! (.some 4) .none (.some (-1))).sliceList! [0, 1, 2, 3, 4] == [4, 3, 2, 1, 0]
#guard (make! (.some 5) .none (.some (-1))).sliceList! [0, 1, 2, 3, 4] == [4, 3, 2, 1, 0]
#guard (make! (.some 500) .none (.some (-1))).sliceList! [0, 1, 2, 3, 4] == [4, 3, 2, 1, 0]
#guard (make! .none .none (.some 2)).sliceList! [0, 1, 2, 3, 4] == [0, 2, 4]
#guard (make! (.some 1) .none (.some 2)).sliceList! [0, 1, 2, 3, 4] == [1, 3]
#guard (make! .none .none (.some (-1))).sliceList! [0, 1, 2, 3, 4] == [4, 3, 2, 1, 0]
#guard (make! .none .none (.some (-2))).sliceList! [0, 1, 2, 3, 4] == [4, 2, 0]
#guard (make! .none .none (.some (-200))).sliceList! [0, 1, 2, 3, 4] == [4]

/-
Iteration over slices is finicky because
1. We can go forwards and backwards
2. We must know when to stop
3. `Nat` is the only type that makes sense for indices into an array.
4. We can start with an empty iterator
-/
structure Iter where
  private mk::
  dim : Nat
  size : Nat
  private start : Nat
  private stop : Option Nat
  private step : Int
  private stepNz : step ≠ 0
  peek : Nat
deriving Repr

instance : Inhabited Iter where
  default := Iter.mk 1 1 0 none 1 (by simp) 0

namespace Iter

private def dir' (n : Int) : Dir := if 0 <= n then Dir.Forward else Dir.Backward

def dir (iter : Iter) : Dir := dir' iter.step

-- The only complexity here is deciding whether the iterator starts off empty
-- so we can set `peek` correctly
def make (slice : Slice) (dim : Nat) : Err Iter :=
  let size := slice.size dim
  -- Can't use `defaults` here because we need the proof that step ≠ 0 and we'd
  -- need dependent pattern matching.
  let start := slice.startOrDefault dim
  let stop := slice.stopOrDefault dim
  let step := slice.stepOrDefault
  let stepNz : step ≠ 0 := slice.stepOrDefaultNz
  let peek := match dir' step, stop with
  | .Forward, _ => if start < stop.getD dim then some start else .none
  | .Backward, .none => .some start -- even if start is 0 we get the 0th element
  | .Backward, .some stop => if stop < start then .some start else .none
  match peek with
  | none => .error "Empty iterator"
  | some peek => .ok { dim, size, start, stop, step, stepNz, peek }

def make! (slice : Slice) (dim : Nat) : Iter := get! $ make slice dim

def next (iter : Iter) : Option Iter := match iter.dir with
| .Forward =>
  let stop := iter.stop.getD iter.dim -- we could show that stop.isSome by `stopOrDefaultForward` above so the default is never used
  let c := (iter.peek + iter.step).toNat
  if stop <= c then none else some { iter with peek := c }
| .Backward =>
  let stop := iter.stop.getD 0
  let c := iter.peek + iter.step
  if c < stop then none else some { iter with peek := c.toNat }

def reset (iter : Iter) : Iter := { iter with peek := iter.start }

instance iteratorInstance : Iterator Iter Nat where
  next := next
  peek := peek
  reset := reset
  size := size

def toList (iter : Iter) : List Nat := iteratorInstance.toList iter

#guard (Iter.make! Slice.all 5).toList == [0, 1, 2, 3, 4]
#guard (Iter.make! (Slice.make! .none .none (.some 2)) 5).toList == [0, 2, 4]
#guard (Iter.make! (Slice.make! (.some 3) .none .none) 5).toList == [3, 4]
#guard (Iter.make! (Slice.make! .none .none (.some (-1))) 5).toList == [4, 3, 2, 1, 0]

end Iter
end Slice
end TensorLib
