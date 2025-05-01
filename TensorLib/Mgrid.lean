/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/

import TensorLib.Common
import TensorLib.Index
import TensorLib.Iterator
import TensorLib.Tensor
import TensorLib.Slice

namespace TensorLib
namespace Mgrid

abbrev elementType : Type := UInt64

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
abbrev MgridIter := Iterator.BEList Iterator.IntIter

namespace MgridIter

def ndim (iter : MgridIter) : Nat := iter.val.length

def make (iters : List Iterator.IntIter) : MgridIter := Iterator.BEList.make iters

def componentSize (iter : MgridIter) : Nat := iter.val.foldl (fun acc iter => acc * iter.size) 1

def tensorSize (iter : MgridIter) : Nat := iter.componentSize * iter.ndim

def toList (iter : MgridIter) : List (List Int) := Iterator.toList iter

#guard
  let iter := Iterator.IntIter.make! 0 3 1
  (make [iter, iter]).toList ==
    [[0, 0], [0, 1], [0, 2],
     [1, 0], [1, 1], [1, 2],
     [2, 0], [2, 1], [2, 2]]

#guard
  let iter0 := Iterator.IntIter.make! 0 2 1
  let iter1 := Iterator.IntIter.make! 4 0 (-1)
  (make [iter0, iter1]).toList ==
  [[0, 4], [0, 3], [0, 2], [0, 1],
   [1, 4], [1, 3], [1, 2], [1, 1]]

#guard
  let iter0 := Iterator.IntIter.make! (-4) (-2) 1
  let iter1 := Iterator.IntIter.make! 7 4 (-1)
  (MgridIter.make [iter0, iter1]).toList == [
    [-4, 7], [-4, 6], [-4, 5],
    [-3, 7], [-3, 6], [-3, 5]
  ]

-- Since there is no dimension, figuring out defaults is tricky. Just fail
-- if it's not obvious.
private def sliceToIntIter (slice : Slice) : Err Iterator.IntIter :=
  let step := slice.step.getD 1
  match slice.start, slice.stop with
  | some start, some stop => Iterator.IntIter.make start stop step
  | _, _ => .error "Can't convert slice to int iterator"

end MgridIter
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
section Mgrid
open scoped Iterator.PairLockStep

def mgrid (slices : List Slice) : Err Tensor := do
  let sliceCount := slices.length
  let iters <- slices.mapM Mgrid.MgridIter.sliceToIntIter
  let slicesDims := iters.map (Iterator.size Int)
  let shape := Shape.mk $ sliceCount :: slicesDims
  let slicesShape := Shape.mk slicesDims
  let dtype := Dtype.int64 -- We fix a little-endian uint64 by convention
  let mut arr := Tensor.zeros dtype shape
  let mgridIter := Mgrid.MgridIter.make iters
  let indexIter := slicesShape.belist
  if Iterator.size (List Int) mgridIter != Iterator.size (List Nat) indexIter then throw "Invariant failure: iterator size mismatch at start"
  for (index, values) in (indexIter, mgridIter) do
    if values.length != sliceCount then .error "Invariant failure: value length mismatch"
    for (i, v) in (List.range sliceCount).zip values do
      let value <- Dtype.int64.byteArrayOfInt v
      arr <- arr.setDimIndex (i :: index) value
  return arr

def mgrid! (slices : List Slice) : Tensor := get! $ mgrid slices

end Mgrid
section Test

open Tensor.Format

private def mg (slices : List Slice) : Tree Int := (mgrid! slices).toIntTree!

-- 0D
#guard mg [] == .root []

-- 1D
#guard mg [Slice.ofStartStop (-4) (-2)] == .node [.root [-4, -3]]
#guard mg [Slice.ofStartStop (-4) 2] == .node [.root [-4, -3, -2, -1, 0, 1]]
#guard mg [Slice.make! (some 2) (some (-4)) (some (-1))] == .node [.root [2, 1, 0, -1, -2, -3]]

-- 2D
#guard (mg [Slice.ofStartStop 2 4, Slice.ofStartStop 4 7]) ==
  .node [
    .node [
      .root [2, 2, 2], .root [3, 3, 3]
    ],
    .node [
      .root [4, 5, 6], .root [4, 5, 6]
    ]
 ]

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

#guard (mgrid! [Slice.ofStartStop 2 4, Slice.ofStartStop 4 7, Slice.ofStop 2]).toIntTree! ==
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

end Test
end TensorLib
