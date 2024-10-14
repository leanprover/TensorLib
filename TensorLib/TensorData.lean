/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/

import Mathlib.Tactic
import TensorLib.NonEmptyList
import TensorLib.TensorElement

namespace TensorLib

--namespace tensor_data

/-!
This file defines TensorData, the raw data underlying a Tensor.
We are representing it today as nested linked lists. We predict
this will be easier to reason about, but will be slow to evaluate.
We intend to have a more compact and faster-to-access representation
in addition to this one in the future for execution.
-/


-- TODO: do sth better than revert
-- TODO: compare with HLO https://openxla.org/stablehlo/spec#constants

abbrev Shape := List Nat

def shapeCount (s: Shape): Nat := match s with
| [] => 0
| [x] => x
| x :: xs => x * shapeCount xs

variable (a: Type)

inductive TensorData a where
| td_root (xs: List a)
| td_node (xs: NonEmptyList (TensorData a))
deriving Inhabited, Repr

namespace TensorData
-- I tried making this implicit with {} but I needed to add @ everwhere
-- and give it the argument anyway. Maybe there is a way to infer at least
-- within a section

def td_node! (xs: List (TensorData a)): TensorData a :=
  .td_node (NonEmptyList.fromList! xs)

-- TODO: Make tail-recursive, perhaps when Lean will eliminite tail calls in mutual-recursion
def hasShape (x: TensorData a) (shape: Shape) : Bool :=
  match x, shape with
  | .td_root y, n :: [] => y.length == n
  | .td_node ys, n :: shape' =>
      ys.length == n &&
      -- `attach` is used in proofs below, e.g. HashShapeOk
      List.all ys.toList.attach (fun ⟨ y, _ ⟩ => hasShape y shape')
  | _, _ => false

-- Inversion lemmas for each case of the pattern match. There is probably a better way to do this.
private lemma hasShapeNil (x: TensorData a): !(x.hasShape a []) := by
  simp [TensorData.hasShape]

private lemma hasShapeRoot (xs: List a) (H: (TensorData.td_root xs).hasShape a (c :: cs)) :
  cs = [] ∧ List.length xs = c := by
  revert H
  unfold TensorData.hasShape
  cases cs
  all_goals simp

private lemma hasShapeNode (H: (TensorData.td_node ys).hasShape a (c :: cs)) :
  ys.length == c ∧ ys.all (fun y => y.hasShape a cs) := by
  revert H
  simp [TensorData.hasShape]
  intro H
  rw [H]
  simp [NonEmptyList.all, NonEmptyList.toList]

-- Inductive version. This is an exercise to get familiar with the pros and cons of
-- functions vs relations in Lean
inductive HasShape : (TensorData a) -> Shape -> Prop where
| HasShapeRoot xs n :
  List.length xs == n
  -> HasShape (TensorData.td_root xs) [n]
| HasShapeNode xs s :
  (∀ x', x' ∈ xs.toList -> HasShape x' s)
  -> HasShape (TensorData.td_node xs) (xs.length :: s)

-- TODO: make decidable instance

theorem HashShapeOk : ∀ (x: TensorData a) (s: Shape),
  x.HasShape a s <-> x.hasShape a s := by
  intro x1 s1
  constructor
  . intro H
    induction H with
    | HasShapeRoot _ _ H =>
      exact H
    | HasShapeNode xs s H1 H2 =>
      unfold TensorData.hasShape
      simp
      exact H2
  . revert x1
    induction s1 with
    | nil =>
      intro x H
      have H1 := hasShapeNil a x
      rw [H] at H1
      trivial
    | cons c cs H1 =>
      intro x
      cases x with
      | td_root xs =>
        intro H1
        have H2 := hasShapeRoot a xs H1
        revert H2
        rw [and_imp]
        intro H2 H3
        rw [H2, <- H3]
        apply (TensorData.HasShape.HasShapeRoot xs xs.length)
        simp
      | td_node xs =>
        intro H2
        have H3 := hasShapeNode a H2
        revert H3
        rw [and_imp]
        simp [NonEmptyList.length, NonEmptyList.all]
        intro H3 H4 H5
        have H6 := (TensorData.HasShape.HasShapeNode xs cs)
        simp [NonEmptyList.length, NonEmptyList.toList, H3] at H6
        apply H6
        . apply (H1 xs.hd)
          exact H4
        . intro y H7
          apply H1
          apply H5
          exact H7

def getShape (x: TensorData a): Shape :=
  match x with
  | .td_root y => [List.length y]
  | .td_node y@(.mk hd _) => y.length :: hd.getShape

-- The `shape` is the dimensions of the tensor.
-- When you know the tensor is well-formed, you can
-- just use `getShape`, which only traverses the first
-- branch
def shapeOpt (x: TensorData a): Option Shape :=
  let s := x.getShape
  if x.hasShape _ s then .some s else .none

#eval shapeOpt _ (.td_root [1, 2, 3])

#eval shapeOpt _ (
  td_node! _ [
    (.td_root [1, 2, 3]),
    (.td_root [1, 2, 3])
  ])

#eval shapeOpt _ (
  td_node! _ [
    (.td_root [1, 2, 3]),
    (.td_root [1, 2])
  ])

-- CPS version of flatten (we're experimenting here, after all)
def flattenAux {b} (x: TensorData a) (k: List a -> b): b :=
  match x with
  | .td_root y => k y
  | (.td_node (.mk hd [])) => flattenAux hd k
  | (.td_node (.mk hd (z :: zs))) => flattenAux hd (fun acc => flattenAux (.td_node (.mk z zs)) (fun acc' => k (acc ++ acc')))

-- Note that `flatten` does not need well-shaped tensors
-- Otherwise should behave like https://numpy.org/doc/2.0/reference/generated/numpy.ndarray.flatten.html
def flattenTailRecursive (x: TensorData a): List a :=
  @flattenAux a (List a) x id

def flatten (x: TensorData a): List a := match x with
| .td_root xs => xs
| .td_node (NonEmptyList.mk y []) => flatten y
| .td_node (NonEmptyList.mk y (z :: zs)) => flatten y ++ flatten (.td_node (.mk z zs))

-- Show we can do CPS and still reason about it relatively easily
private lemma flattenAuxOk (x: TensorData a) (k: List a -> b) : flattenAux _ x k = k (flatten _ x) := by
  cases x with
  | td_root xs =>
    simp [flatten, flattenAux]
  | td_node xs => match H:xs with
    | .mk y [] =>
      simp [flatten, flattenAux]
      exact (flattenAuxOk y k)
    | .mk y ws@(z :: zs) =>
      rename_i H1
      simp [flatten, flattenAux]
      let w := NonEmptyList.mk z zs
      generalize H2 : (fun acc => flattenAux a (td_node w) fun acc' => k (acc ++ acc')) = k'
      rw [flattenAuxOk y k']
      rw [<-H2]
      simp
      generalize H3 : (fun acc' => k (flatten a y ++ acc')) = k''
      rw [flattenAuxOk (.td_node w) k'']
      rw [<-H3]

lemma flattensSame (x: TensorData a) : flatten _ x = flattenTailRecursive _ x := by
  unfold flattenTailRecursive
  have H := flattenAuxOk _ x id
  simp at H
  rw [H]

#eval flatten _ (
  .td_root ([]: List Nat)
)

#eval flatten _ (
  .td_root ([1, 2, 3]: List Nat)
)

#eval flatten _ (
  td_node! _ [
    .td_root ([1, 2, 3]: List Nat),
    .td_root ([4, 5, 6]: List Nat),
    .td_root ([7, 8, 9]: List Nat),
  ]
)

#eval flatten _ (
  td_node! _ [
    .td_root ([1, 2, 3]: List Nat),
    td_node! _ [
      .td_root ([1, 2, 3]: List Nat),
      .td_root ([4, 5, 6]: List Nat),
      .td_root ([7, 8, 9]: List Nat),
    ],
    .td_root ([1, 2, 3]: List Nat),
  ]
)

section Arith
  variable [addI: Add a]

  private def addLists (x: List a) (y: List a): Option (List a) :=
    match x, y with
    | [], [] => some []
    | _, [] => none
    | [], _ => none
    | a :: as, b :: bs => do
      let rest <- addLists as bs
      return (a + b) :: rest

  mutual
    def add (t1: TensorData a) (t2: TensorData a): Option (TensorData a) :=
      match t1, t2 with
      | .td_root v1, .td_root v2 => do
        let rest <- addLists a v1 v2
        return .td_root rest
      | .td_node (.mk xhd xtl), .td_node (.mk yhd ytl) => do
        let u <- add xhd yhd
        let us <- addList xtl ytl
        return .td_node (NonEmptyList.mk u us)
      | _, _ => .none

    def addList (t1: List (TensorData a)) (t2: List (TensorData a)): Option (List (TensorData a)) :=
      match t1, t2 with
      | [], [] => some []
      | x :: xs, y :: ys => do
        let z <- add x y
        let zs <- addList xs ys
        return z :: zs
      | _, _ => none
  end
end Arith

#eval (add _ (
  td_node! _ [
    .td_root ([1, 2, 3]: List Nat),
    .td_root ([4, 5, 6]: List Nat),
    .td_root ([7, 8, 9]: List Nat),
  ]
) (
  td_node! _ [
    .td_root ([1, 2, 3]: List Nat),
    .td_root ([4, 5, 6]: List Nat),
    .td_root ([7, 8, 9]: List Nat),
  ]
))

#eval (add _ (
  td_node! _ [
    .td_root ([1, 2]: List Nat),
  ]
) (
  td_node! _ [
    .td_root ([1, 2, 3]: List Nat)
  ]
))

end TensorData
end TensorLib
