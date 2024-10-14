/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/
namespace TensorLib

variable {a: Type}
variable [BEq a][Inhabited a]

structure NonEmptyList (a: Type) where
  hd: a
  tl: List a
deriving Inhabited

namespace NonEmptyList

def toList (x: NonEmptyList a): List a := x.hd :: x.tl

-- Use the nice list bracket concrete syntax
instance NelRepr [Repr a] : Repr (NonEmptyList a) where
  reprPrec x n := reprPrec x.toList n

def length (x: NonEmptyList a): Nat := 1 + x.tl.length

def all (x: NonEmptyList a) (P: a -> Bool): Bool :=
  P x.hd && x.tl.all P

def contains (x: NonEmptyList a) (y: a): Bool := (y == x.hd) || x.tl.contains y

-- For examples
protected def fromList! (x: List a): NonEmptyList a := match x with
| [] => panic "empty list"
| x :: xs => NonEmptyList.mk x xs

end NonEmptyList
