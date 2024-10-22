/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: John Tristan, Paul Govereau, Sean McLaughlin
-/

namespace TensorLib

-- Since one goal of TensorLib is to support ML frameworks and languages
-- like Triton and NKI, we will need to support a menagerie of data types.
-- Triton and NKI supports signed and unsigned ints (8/16/32/64) and
-- several floating point types and sizes (float, bfloat, tfloat).
class TensorElement (a: Type) [Add a][Sub a][Mul a][Neg a] /- etc. -/ where
  -- Implement stuff here that's needed for numpy-like ops

-- Abritrary precision
instance : TensorElement Int where

-- 64 bit IEEE-754 floats
instance : TensorElement Float where

-- Code for 2s compliment ints is here https://gist.github.com/ammkrn/79281ae3d3b301c99a84821c18dcb5f1
-- TODO: Int8/16/32/64

-- TODO: Figure out unsigned ints
-- TODO: bfloat, tfloat
