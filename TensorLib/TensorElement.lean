/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/

namespace TensorLib

class TensorElement (a: Type) [Add a][Sub a][Mul a][Neg a] where

section TensorElement

-- Style of repeating the arguments to a class followed by variables was
-- suggested by Lean office hours folks
variable (a: Type)
variable [Add a][Sub a][Mul a][Neg a]

-- ...Lots of definitions here using the variables above...

end TensorElement

instance : TensorElement Int where

-- 64 bit IEEE-754 floats
instance : TensorElement Float where

-- TODO: Code for 2s compliment ints is here https://gist.github.com/ammkrn/79281ae3d3b301c99a84821c18dcb5f1
-- TODO: Int8/16/32/64

-- TODO: Figure out unsigned ints
-- TODO: bfloat, tfloat
