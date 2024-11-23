import TensorLib.NumpyRepr

namespace TensorLib
namespace Slice

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

structure Slice where
  start : Option Int
  stop : Option Int
  step : Option Int
  StepNz : step != .some 0
deriving BEq

inductive Dir where
| Forward
| Backward

-- A slice where we know the size of the array/dimension
structure DimSlice where
  dim : Nat
  dir : Dir
  start : Nat
  stop : Nat
  step : Nat
  StepNz : step != 0
  -- A start, stop, or step of `dim` has the same effect as anything larger than `dim`
  StartOk : start <= dim
  StopOk : stop <= dim
  StepOk : step <= dim

namespace Slice

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
-/
def toDimSlice (slice : Slice) (dim : ℕ) : DimSlice :=
  let (dir, step) := match H: slice.step with
  | .none => (Dir.Forward, 1)
  | .some 0 =>
    let t : False := by
      let H1 := slice.StepNz
      rw [H] at H1
      trivial
    nomatch t
  | .some n =>
    let (dir, n) := if 0 < n then (Dir.Forward, n) else (Dir.Backward, -n)
    (dir, min dim n.toNat)
  let start := match slice.start, dir with
  | .none, .Forward => 0
  | .none, .Backward => dim - 1
  | .some n, _ => if n <= 0 then 0 else min dim n.toNat
  let stop := match slice.stop, dir with
  | .none, .Forward => dim
  | .none, .Backward => 0
  | .some n, _ => if n <= 0 then 0 else min dim n.toNat
  let StepNz := sorry
  let StartOk := sorry
  let StopOk := sorry
  let StepOk := sorry
  DimSlice.mk dim dir start stop step StepNz StartOk StopOk StepOk


def isEmpty (slice : Slice) :=
  (0 < slice.step && slice.stop <= slice.start) ||
  (slice.step < 0 && slice.start <= slice.stop)

def size (slice : Slice) : ℕ := (slice.stop - slice.start) / slice.step.natAbs

private def oneNz : 1 != 0 := by trivial
def fromNat (n : Nat) : Slice := Slice.mk n (n+1) 1
def fromStop (n : Nat) : Slice := Slice.mk 0 n 1
def fromStartStop (start stop : Nat) : Slice := Slice.mk start stop 1


private def nonneg (n : Int) (dim : Nat) : Err Nat :=
  if n < 0 then
    if dim < -n then .error s!"index {n} out of bounds {dim}" else .ok (dim - n).toNat
  else
    if dim < n then .error s!"index {n} out of bounds {dim}" else .ok n.toNat

-- Get rid of negative and too-large values
def normalize (slice : NumpySlice) (dim : Nat) : Err Slice := do
  let start <- nonneg slice.start dim
  let stop <- nonneg slice.stop dim
  if slice.step == 0 then .error "step can not be 0"
  else .ok (Slice.mk start stop slice.step)

end Slice
end TensorLib
