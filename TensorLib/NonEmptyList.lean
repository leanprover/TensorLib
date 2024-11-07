/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/

import Mathlib.Tactic

/-!
# Nonempty lists
-/
namespace TensorLib

variable {a: Type}

structure NonEmptyList (a: Type) where
  hd: a
  tl: List a
  deriving BEq, Inhabited

namespace NonEmptyList

def toList (x: NonEmptyList a): List a := x.hd :: x.tl

instance : Coe (NonEmptyList α) (List α) where
  coe := toList

instance : CoeDep (List α) (x :: xs) (NonEmptyList α) where
  coe := { hd := x, tl := xs }

-- Use the nice list bracket concrete syntax
instance NelRepr [Repr a] : Repr (NonEmptyList a) where
  reprPrec x n := reprPrec x.toList n

def length (x: NonEmptyList a): Nat := 1 + x.tl.length

@[simp]
def append (x : NonEmptyList a) (y : NonEmptyList a) : NonEmptyList a :=
  NonEmptyList.mk x.hd (x.tl ++ y.toList)

theorem appendLength (x y : NonEmptyList a) : (append x y).length = x.length + y.length := by
  simp [length, toList]
  linarith

instance : HAppend (NonEmptyList a) (NonEmptyList a) (NonEmptyList a) where
  hAppend := append

theorem hAppendLength (x y : NonEmptyList a) : (x ++ y).length = x.length + y.length := by
  apply appendLength

def appendListL (x : List a) (y : NonEmptyList a) := match x with
| [] => y
| x :: xs => NonEmptyList.mk x (xs ++ y.toList)

theorem appendListLLength (x : List a) (y : NonEmptyList a) : (appendListL x y).length = x.length + y.length := by
  induction x
  . simp [appendListL]
  . simp [appendListL, NonEmptyList.length, NonEmptyList.toList]
    linarith

instance HAppendListL : HAppend (List a) (NonEmptyList a) (NonEmptyList a) where
  hAppend := appendListL

theorem hAppendListLLength (x : List a) (y : NonEmptyList a) : (appendListL x y).length = x.length + y.length := by
  apply appendListLLength

def appendListR (x : NonEmptyList a) (y : List a) := NonEmptyList.mk x.hd (x.tl ++ y)

instance HAppendListR : HAppend (NonEmptyList a) (List a) (NonEmptyList a) where
  hAppend := appendListR

theorem appendListRLength (x : NonEmptyList a) (y : List a) : (appendListR x y).length = x.length + y.length := by
  simp [appendListR, NonEmptyList.length]
  linarith

theorem hAppendListRLength (x : NonEmptyList a) (y : List a) : (appendListR x y).length = x.length + y.length := by
  apply appendListRLength

def all (x: NonEmptyList a) (P: a -> Bool): Bool :=
  P x.hd && x.tl.all P

def contains [BEq a] (x : NonEmptyList a) (y : a) : Bool := (y == x.hd) || x.tl.contains y

def map {b : Type} (f : a -> b) (x : NonEmptyList a) : NonEmptyList b :=
  NonEmptyList.mk (f x.hd) (List.map f x.tl)

def zipWith {b c : Type} (f : a -> b -> c) (x : NonEmptyList a) (y : NonEmptyList b) : NonEmptyList c :=
  NonEmptyList.mk (f x.hd y.hd) (List.zipWith f x.tl y.tl)

def zip {b : Type} (x : NonEmptyList a) (y : NonEmptyList b) : NonEmptyList (a × b) :=
  zipWith (fun x y => (x, y)) x y

def foldl {b : Type} (f : a -> b -> a) (x : a) (xs : NonEmptyList b) : a :=
  List.foldl f (f x xs.hd) xs.tl

def foldr {b : Type} (f : a -> b -> b) (x : b) (xs : NonEmptyList a) : b :=
  List.foldr f (f xs.hd x) xs.tl

def traverse [Applicative F] (f : α → F β) (x: NonEmptyList α) : F (NonEmptyList β) :=
   NonEmptyList.mk <$> f x.hd <*> List.traverse f x.tl

def reverse (x : NonEmptyList a) : NonEmptyList a :=
  match x.tl.reverse with
  | [] => x
  | y :: ys => { hd := y, tl := ys ++ [x.hd] }

#eval reverse { hd := 5, tl := [1, 2, 3] }

instance NonEmptyListFunctor : Functor NonEmptyList where
  map := map

instance NonEmptyListTraversable : Traversable NonEmptyList where
  traverse := traverse

-- For examples
section Unsafe

variable [Inhabited a]

protected def fromList! (x: List a): NonEmptyList a := match x with
| [] => panic "empty list"
| x :: xs => NonEmptyList.mk x xs

end Unsafe

end NonEmptyList
