/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/

import Init.System.IO
import TensorLib.Tensor

namespace TensorLib

section Test

-- Caller must remove the temp file
private def saveNumpyArray (expr : String) : IO System.FilePath := do
  let (_root_, file) <- IO.FS.createTempFile
  let expr := s!"import numpy as np; x = {expr}; np.save('{file}', x)"
  let _output <- IO.Process.output { cmd := "/usr/bin/env", args := ["python3", "-c", expr].toArray }
  -- `np.save` appends `.npy` to the file
  return file.addExtension "npy"

private def testTensorElementBV (dtype : Dtype) := do
  let file <- saveNumpyArray s!"np.arange(20, dtype='{dtype.name}').reshape(5, 4)"
  let npy <- Npy.parseFile file
  let arr <- IO.ofExcept (Tensor.ofNpy npy)
  let _ <- IO.FS.removeFile file
  let expected := (Tensor.arange! dtype 20).reshape! (Shape.mk [5, 4])
  return Tensor.arrayEqual expected arr

-- Sketchy perhaps, but seems to work for testing
private def runIo [Inhabited a] (x : IO a) : a := match x.run () with
| .error _ _ => default
| .ok v _ => v

#guard runIo (testTensorElementBV Dtype.uint16)
#guard runIo (testTensorElementBV Dtype.uint32)

end Test
