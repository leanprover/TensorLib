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
import TensorLib.Shape

/-!
Broadcasting is a convenience and performance trick to allow operations that expect the same
shaped arguments to work on non-matching arguments.  For example, we would like to be able
to add 1 to each element of a tensor without building the all-1s tensor in memory.
It involves applying the following rules to two tensors

1. If the shape of one is smaller than the other, pad the smaller one
   with 1s until they are the same length
2. For each pair of numbers at each index, to broadcast either the
   numbers must be the same, or one of them should be 1. In the later
   case we replace that shape with the other number

For example, we broadcast (3, 2, 1) (2, 7) to (3, 2, 7).

A: (3, 2, 1)
B: (2, 7)

Rule 1

A: (3, 2, 1)
B: (1, 2, 7)

Rule 2

A: (3, 2, 1)
B: (3, 2, 7)

Rule 2

A: (3, 2, 7)
B: (3, 2, 7)

Theorem to prove: If we can broadcast s1 to s2, then given an array with shape s1, then s1.reshape s2 succeeds
-/

namespace TensorLib

structure Broadcast where
  left : Shape
  right : Shape
  deriving BEq, Repr

namespace Broadcast

-- In broadcasting, we first extend the shorter array by prefixing 1s.
-- NKI semantics currently suffixes 1s in some cases, so be explicit about
-- the naming.
private def oneExtendPrefix (b : Broadcast) : Broadcast :=
  let n1 := b.left.ndim
  let n2 := b.right.ndim
  if n1 <= n2
  then { b with left := Shape.mk $ List.replicate (n2 - n1) 1 ++ b.left.val }
  else { b with right := Shape.mk $ List.replicate (n1 - n2) 1 ++ b.right.val }

private theorem oneExtendPrefixLength (b : Broadcast) :
  let b' := oneExtendPrefix b
  b'.left.ndim = b'.right.ndim := by
  cases b
  rename_i left right
  simp [oneExtendPrefix]
  by_cases H : left.ndim <= right.ndim
  . simp_all [Shape.ndim, H]
  . simp_all [Shape.ndim, H]
    aesop (config := { warnOnNonterminal := false })
    rw [Nat.sub_add_cancel]
    omega

private def matchPairs (b : Broadcast) : Option Shape :=
  if b.left.ndim != b.right.ndim then none else
  let f xy := match xy with
    | (x, y) =>
      if x == y then some x
      else if x == 1 then some y
      else if y == 1 then some x
      else none
  let dims := (b.left.val.zip b.right.val).mapM f
  dims.map Shape.mk

/-
Returns the shape resulting from broadcast the arguments.

Sometimes you want to fix one side of the broadcast, and only allow the other to extend/expand.
An example of this is `arr.broadcastTo (2, 3, 4)` where we really want the array to have the
shape `(2,3,4)`, not some compatible shape like `(2, 2, 3, 4)` if, say, `arr.shape = (2, 1, 1, 1)`.
We don't need a separate function for this because you can just compare the fixed side to the
solution and ensure they are the same. They will be equal iff the broadcasting didn't impact the
fixed side.
-/
def broadcast (b : Broadcast) : Option Shape := matchPairs (oneExtendPrefix b)

--! Whether broadcasting is possible
def canBroadcast (b : Broadcast) : Bool := (broadcast b).isSome

#guard matchPairs (Broadcast.mk (Shape.mk [1, 2, 3]) (Shape.mk [7, 2, 1])) == some (Shape.mk [7, 2, 3])
#guard broadcast (Broadcast.mk (Shape.mk [1, 2, 3]) (Shape.mk [7, 7, 9, 2, 1])) == some (Shape.mk [7, 7, 9, 2, 3])

-- todo: add plausible properties when property-based testing settles down in Lean-land
#guard
 let x1 := Shape.mk [1, 2, 3]
 let x2 := Shape.mk [2, 3]
 let b1 := Broadcast.mk x1 x1
 let b2 := Broadcast.mk x1 x2
 oneExtendPrefix b1 == b1 &&
 broadcast b2 == broadcast b1 &&
 broadcast b2 == .some (Shape.mk [1, 2, 3])

def broadcastList (shapes : List Shape) : Option Shape := Id.run do
  match shapes with
  | [] => none
  | shape :: shapes =>
  let mut shape := shape
  for s in shapes do
    let b := Broadcast.mk shape s
    match b.broadcast with
    | .none => return .none
    | .some s =>
      shape := s
  return shape

#guard
 let x1 := Shape.mk [1, 2, 3]
 let x2 := Shape.mk [2, 3]
 let x3 := Shape.mk []
 broadcastList [x1, x2, x3] == .some x1

end Broadcast
end TensorLib
