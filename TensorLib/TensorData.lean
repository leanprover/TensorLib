/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: John Tristan, Paul Govereau, Sean McLaughlin
-/
import Mathlib.Tactic
import TensorLib.TensorElement

namespace TensorLib

/-!
This file defines TensorData, the raw data underlying a Tensor.
We are representing it today as nested linked lists. We predict
this will be easier to reason about, but will be slow to evaluate.
We intend to have a more compact and faster-to-access representation
in addition to this one in the future for execution.
-/

-- I tried making this implicit with {} but I needed to add @ everwhere
-- and give it the argument anyway. Maybe there is a way to infer at least
-- within a section
variable (a: Type)

-- I wonder if writing all these out will get old. Can Lean infer the Add/Sub/etc.
-- somehow?
variable [addI: Add a][subI: Sub a][mulI: Mul a][divI: Div a][negI: Neg a]
         [tensorElementI: TensorElement a]

-- I tried `td_node (x: TensorData) (xs: List TensorData)` to remove the redundant empty node
-- from the type. (We can already represent the 0 shape with `td_root []`.)
-- This design works, but this way the examples are more uniform; it looked weird to
-- syntactically preference the first of every node list. The code has some annoying special
-- cases for the empty node now, though, so it may be worth revisiting this.
inductive TensorData where
| td_root (xs: List a)
| td_node (xs: List TensorData)

abbrev Shape := List Nat

-- TODO: Make tail-recursive, perhaps when Lean will eliminite tail calls in mutual-recursion
  def TensorData.hasShape (x: TensorData a) (shape: Shape): Bool :=
    match x, shape with
    | .td_root y, List.cons n .nil => List.length y == n
    | .td_node ys, List.cons n shape' =>
        List.length ys == n &&
        -- `attach` is used in proofs below, e.g. HashShapeOk
        List.all ys.attach (fun ⟨ y, _ ⟩ => hasShape y shape')
    | _, _ => false

inductive TensorData.HasShape : (TensorData a) -> Shape -> Prop where
| HasShapeRoot xs n : List.length xs == n -> HasShape (TensorData.td_root xs) [n]
| HasShapeNil : HasShape (TensorData.td_root []) [0]
| HasShapeCons x xs s : HasShape x s
   -> (∀ x', x' ∈ xs  -> HasShape x' s)
   -> HasShape (TensorData.td_node (.cons x xs)) (.cons (1 + List.length xs) s)

theorem HashShapeOk : ∀ (x: TensorData a) (s: Shape), x.HasShape a s <-> x.hasShape a s := by
  intro x1 s1
  constructor
  . intro H
    induction H with
    | HasShapeRoot xs n H =>
      unfold TensorData.hasShape
      exact H
    | HasShapeNil =>
      unfold TensorData.hasShape
      simp
    | HasShapeCons x xs s H1 H2 H3 H4 =>
      clear x1 s1
      unfold TensorData.hasShape
      simp [H3]
      constructor
      . linarith
      . intros y H5 y' H6 H7
        apply H4
        rw[<- H7]
        exact H6
  . revert s1
    induction x1


--  cases x
--  unfold TensorData.hasShape

def TensorData.getShape (x: TensorData a): Shape :=
  match x with
  | .td_root y => [List.length y]
  | .td_node ys => match ys with
    | .nil => [0]
    | .cons y _ => .cons (List.length ys) (getShape y)

-- The `shape` is the dimensions of the tensor.
-- When you know the tensor is well-formed, you can
-- just use `getShape`, which only traverses the first
-- branch
def shapeOpt (x: TensorData a): Option Shape :=
  let s := x.getShape
  if x.hasShape _ s then .some s else .none

#eval shapeOpt _ (.td_root [1, 2, 3])

#eval shapeOpt _ (
  .td_node [
    (.td_root [1, 2, 3]),
    (.td_root [1, 2, 3])
  ])

#eval shapeOpt _ (
  .td_node [
    (.td_root [1, 2, 3]),
    (.td_root [1, 2])
  ])


-- CPS version of flatten (we're experimenting here, after all)
def flattenAux {b} (x: TensorData a) (k: List a -> b): b :=
  match x with
  | .td_root x => k x
  | .td_node .nil => k []
  | .td_node (.cons y ys) =>
    flattenAux y (fun acc => flattenAux (.td_node ys) (fun acc' => k (List.append acc acc')))

-- Note that `flatten` does not need well-shaped tensors
-- Otherwise should behave like https://numpy.org/doc/2.0/reference/generated/numpy.ndarray.flatten.html
def flatten (x: TensorData a): List a :=
  @flattenAux a (List a) x id

-- Mutually recursive version of flatten. This is not tail-recursive, but let's
-- consider it as an option if the proofs go through easier with it.
mutual
  def flatten_mutual (x: TensorData a): List a := match x with
    | .td_root ys => ys
    | .td_node xs => flatten_list xs

  def flatten_list (xs: List (TensorData a)): List a := match xs with
    | .nil => []
    | .cons d ds => List.append (flatten_mutual d) (flatten_list ds)
end

#eval flatten _ (
  .td_root ([]: List Nat)
)

#eval flatten _ (
  .td_root ([1, 2, 3]: List Nat)
)

#eval flatten _ (
  TensorData.td_node [
    .td_root ([1, 2, 3]: List Nat),
    .td_root ([4, 5, 6]: List Nat),
    .td_root ([7, 8, 9]: List Nat),
  ]
)

#eval flatten _ (
  .td_node [
    .td_root ([1, 2, 3]: List Nat),
    .td_node [
      .td_root ([1, 2, 3]: List Nat),
      .td_root ([4, 5, 6]: List Nat),
      .td_root ([7, 8, 9]: List Nat),
    ],
    .td_root ([1, 2, 3]: List Nat),
  ]
)

theorem flattens_same : ∀ x, flatten a x = flatten_mutual a x := by
  intro x
  cases x
  . unfold flatten flattenAux flatten_mutual
    simp
  .

    -- unfold flatten_mutual flatten_list



end TensorData
end TensorLib
