import Aesop
import TensorLib.Common

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
> Then, if i is not given it defaults to 0 for step > 0 and n - 1 for step < 0 .
> If j is not given it defaults to n for k > 0 and -n-1 for k < 0 .
> If k is not given it defaults to 1.
> Negative values are allowed. The meaning of x[-k] is x[n-k] where `n` is the size of the relevant dimension
>
> Note that these requirements from the NumPy docs don't make sense if `n = 0`.
-/
structure Slice where
  start : Option Int
  stop : Option Int
  step : Option Int
  stepNz : step ≠ .some 0
deriving BEq

instance : Inhabited Slice where
  default := Slice.mk .none .none .none (by simp)

namespace Slice

def build (start stop step : Option Int) : Err Slice :=
  match H : step with
  | .none =>
    let stepNz : step ≠ some 0 := by aesop
    .ok (Slice.mk start stop step stepNz)
  | .some k =>
    if H1 : k == 0 then .error "step can't be 0" else
    let stepNz : step ≠ some 0 := by aesop
    .ok (Slice.mk start stop step stepNz)

private partial def build! (start stop step : Option Int) : Slice :=
  get! (build start stop step)

def dir (s : Slice): Dir := match s.step with
  | .none => Dir.Forward
  | .some k => if k < 0 then Dir.Backward else Dir.Forward

/-
Calculating the nat start index of a slice is complicated by the
NumPy defaults that differ based on direction (forward/backward)
and negative values.
-/
def startOrDefault (s : Slice) (n : Nat) : Nat :=
  let dir := s.dir
  match s.start with
  | .none => match dir with
    | Dir.Forward => 0
    | Dir.Backward => n - 1
  | .some k =>
    if k < -n then 0
    else if -n <= k && k < 0 then (n + k).toNat
    else if 0 <= k && k < n then k.toNat
    else /- n <= k -/ match dir with
    | Dir.Forward => n
    | Dir.Backward => n - 1

/-
We use an option here to signal that the stopping point for negative step
is the first element of the list. We can't use 0 because the stopping point
is exclusive in a slice. [start ... stop) We can't use any negative number
because the first `n` negative numbers wrap back around to positive numbers.
-/
def stopOrDefault (s : Slice) (n : Nat) : Option Nat :=
  match s.stop, s.dir with
  | .none, .Forward => .some n
  | .none, .Backward => .none
  | .some k, .Forward =>
    if k < -n then .some 0
    else if -n <= k && k < 0 then .some (n + k).toNat
    else if 0 <= k && k < n then .some k.toNat
    else /- n <= k -/ .some n
  | .some k, .Backward =>
    if k < -n then .none
    else if -n <= k && k < 0 then .some (n + k).toNat
    else if 0 <= k && k < n then .some k.toNat
    else /- n <= k -/ .some n

theorem stopRange (s : Slice) (n : Nat) :
  match s.stopOrDefault n with
  | .none => True
  | .some k => k <= n
:= by
  unfold stopOrDefault
  cases s.stop
  . cases s.dir
    . simp
    . simp
  . rename_i k
    cases s.dir
    . simp
      by_cases H : k < -n
      all_goals simp [H]
      by_cases H1 : k < 0
      . simp [H, H1]
        simp_all only [not_lt, ↓reduceIte]
        all_goals omega
      . aesop (config := { warnOnNonterminal := false })
        all_goals omega
    . simp
      by_cases H : k < -n
      all_goals simp [H]
      by_cases H1 : k < 0
      . simp [H, H1]
        aesop (config := { warnOnNonterminal := false })
        all_goals omega
      . aesop (config := { warnOnNonterminal := false })
        all_goals omega

theorem stopForward (s : Slice) (dim : Nat) (H : s.dir = .Forward) : (s.stopOrDefault dim).isSome := by
  unfold stopOrDefault
  aesop

def stepOrDefault (s : Slice) : Int := s.step.getD 1

theorem stepForward (s : Slice) (H : s.dir = .Forward) : 0 < s.stepOrDefault := by
  revert H
  unfold stepOrDefault dir
  cases H1 : s.step
  all_goals simp
  rename_i k
  have H2 := s.stepNz
  aesop
  omega

def defaults (s : Slice) (dim : Nat) : Nat × Option Nat × Int :=
  (s.startOrDefault dim, s.stopOrDefault dim, s.stepOrDefault)

def size (s : Slice) (dim : Nat) : Nat :=
  -- Can't use `defaults` here because I need equalities below which don't work with tuple splits
  let start := s.startOrDefault dim
  let stop := s.stopOrDefault dim
  let step := s.stepOrDefault
  match H1 : s.dir, H2 : stop with
  | .Forward, .none =>
    let k : False := by
      have H2 := s.stopForward dim H1
      aesop
    nomatch k
  | .Forward, .some stop => (stop - start) / step.toNat
  | .Backward, .none => (start + 1) / step.natAbs
  | .Backward, .some stop => (start - stop) / step.natAbs

#guard (Slice.build! .none .none .none).size 10 == 10
#guard (Slice.build! .none .none (.some (-1))).size 10 == 10
#guard (Slice.build! .none .none (.some (-2))).size 10 == 5
#guard (Slice.build! .none .none (.some (2))).size 10 == 5
#guard (Slice.build! (.some 5) .none .none).size 10 == 5
#guard (Slice.build! (.some 5) .none (.some 1)).size 10 == 5
#guard (Slice.build! (.some 5) .none (.some 3)).size 10 == 1
#guard (Slice.build! (.some 5) .none (.some (-1))).size 10 == 6
#guard (Slice.build! (.some 5) .none (.some (-3))).size 10 == 2
#guard (Slice.build! .none (.some 5) .none).size 10 == 5
#guard (Slice.build! .none (.some 5) (.some (-1))).size 10 == 4

-- Reference implementation. A numpy slice doesn't copy the data, but just updates the
-- start and strides of the ndarray.
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
    loop (xs.get! i.toNat :: acc) (i + step)
  loop [] start

#guard (Slice.build! .none .none .none).sliceList! [0, 1, 2, 3, 4] == [0, 1, 2, 3, 4]
#guard (Slice.build! (.some 0) .none .none).sliceList! [0, 1, 2, 3, 4] == [0, 1, 2, 3, 4]
#guard (Slice.build! (.some 0) (.some 5) .none).sliceList! [0, 1, 2, 3, 4] == [0, 1, 2, 3, 4]
#guard (Slice.build! (.some 0) (.some 5) (.some 1)).sliceList! [0, 1, 2, 3, 4] == [0, 1, 2, 3, 4]
#guard (Slice.build! (.some 3) (.some 5) (.some 1)).sliceList! [0, 1, 2, 3, 4] == [3, 4]
#guard (Slice.build! (.some 10) (.some 5) (.some 1)).sliceList! [0, 1, 2, 3, 4] == []
#guard (Slice.build! (.some (-10)) (.some 5) (.some 1)).sliceList! [0, 1, 2, 3, 4] == [0, 1, 2, 3, 4]
#guard (Slice.build! (.some (-10)) (.some 5) (.some 100)).sliceList! [0, 1, 2, 3, 4] == [0]
#guard (Slice.build! .none .none (.some (-1))).sliceList! [0, 1, 2, 3, 4] == [4, 3, 2, 1, 0]
#guard (Slice.build! .none (.some 0) (.some (-1))).sliceList! [0, 1, 2, 3, 4] == [4, 3, 2, 1]
#guard (Slice.build! .none (.some (-5)) (.some (-1))).sliceList! [0, 1, 2, 3, 4] == [4, 3, 2, 1]
#guard (Slice.build! .none (.some (-6)) (.some (-1))).sliceList! [0, 1, 2, 3, 4] == [4, 3, 2, 1, 0]
#guard (Slice.build! .none (.some (-600)) (.some (-1))).sliceList! [0, 1, 2, 3, 4] == [4, 3, 2, 1, 0]
#guard (Slice.build! (.some 4) .none (.some (-1))).sliceList! [0, 1, 2, 3, 4] == [4, 3, 2, 1, 0]
#guard (Slice.build! (.some 5) .none (.some (-1))).sliceList! [0, 1, 2, 3, 4] == [4, 3, 2, 1, 0]
#guard (Slice.build! (.some 500) .none (.some (-1))).sliceList! [0, 1, 2, 3, 4] == [4, 3, 2, 1, 0]
#guard (Slice.build! .none .none (.some 2)).sliceList! [0, 1, 2, 3, 4] == [0, 2, 4]
#guard (Slice.build! (.some 1) .none (.some 2)).sliceList! [0, 1, 2, 3, 4] == [1, 3]
#guard (Slice.build! .none .none (.some (-2))).sliceList! [0, 1, 2, 3, 4] == [4, 2, 0]
#guard (Slice.build! .none .none (.some (-200))).sliceList! [0, 1, 2, 3, 4] == [4]

end Slice
end TensorLib
