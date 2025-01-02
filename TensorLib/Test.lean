/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/

import Init.System.IO
import TensorLib.Tensor

namespace TensorLib

section Test

private def DEBUG := false
private def debugPrint {a : Type} [Repr a] (s : a) : IO Unit := if DEBUG then IO.print (Std.Format.pretty (repr s)) else return ()
private def debugPrintln {a : Type} [Repr a] (s : a) : IO Unit := do
  debugPrint s
  if DEBUG then IO.print "\n" else return ()

-- Caller must remove the temp file
private def saveNumpyArray (expr : String) : IO System.FilePath := do
  let (_root_, file) <- IO.FS.createTempFile
  let expr := s!"import numpy as np; x = {expr}; np.save('{file}', x)"
  let output <- IO.Process.output { cmd := "/usr/bin/env", args := ["python3", "-c", expr].toArray }
  let _ <- debugPrintln output.stdout
  let _ <- debugPrintln output.stderr
  -- `np.save` appends `.npy` to the file
  return file.addExtension "npy"

private def testTensorElementBV (n : Nat) [Tensor.Element (BitVec n)] (dtype : String) : IO Bool := do
  let file <- saveNumpyArray s!"np.arange(20, dtype='{dtype}').reshape(5, 4)"
  let npy <- Npy.parseFile file
  let arr <- IO.ofExcept (Tensor.ofNpy npy)
  let _ <- debugPrintln file
  let _ <- debugPrintln arr
  let _ <- IO.FS.removeFile file
  let expected : List (BitVec n) := (List.range 20).map (BitVec.ofNat n)
  let mut actual := []
  for i in [0:20] do
    match Tensor.Element.getPosition arr i with
    | .error msg => IO.throwServerError msg
    | .ok v => actual := v :: actual
  actual := actual.reverse
  let _ <- debugPrintln actual
  return expected == actual

-- Sketchy perhaps, but seems to work for testing
private def ioBool (x : IO Bool) : Bool := match x.run () with
| .error _ _ => false
| .ok b _ => b

#guard ioBool (testTensorElementBV 16 "uint16")
#guard ! ioBool (testTensorElementBV 32 "uint16")
#guard ioBool (testTensorElementBV 32 "uint32")
#guard ! ioBool (testTensorElementBV 32 "uint64")

end Test
