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

import Init.System.IO
import TensorLib.Tensor

namespace TensorLib

namespace Test

-- Caller must remove the temp file
private def saveNumpyArray (expr : String) : IO System.FilePath := do
  let (_root_, file) <- IO.FS.createTempFile
  let expr := s!"import numpy as np; x = {expr}; np.save('{file}', x)"
  let _output <- IO.Process.output { cmd := "/usr/bin/env", args := ["python3", "-c", expr].toArray }
  -- `np.save` appends `.npy` to the file
  return file.addExtension "npy"

private def testTensorElementBV (dtype : Dtype) : IO Bool := do
  let file <- saveNumpyArray s!"np.arange(20, dtype='{dtype}').reshape(5, 4)"
  let npy <- Npy.parseFile file
  let arr <- IO.ofExcept (Tensor.ofNpy npy)
  let _ <- IO.FS.removeFile file
  let expected := (Tensor.arange! dtype 20).reshape! (Shape.mk [5, 4])
  return Tensor.arrayEqual expected arr

def runAllTests : IO Bool := do
  return (<- testTensorElementBV Dtype.uint16) &&
         (<- testTensorElementBV Dtype.uint32)

end Test
end TensorLib
