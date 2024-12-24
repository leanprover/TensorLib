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
  let expected : List (BitVec n) := List.range 20
  let actual := do
    let rev <- Nat.foldM (fun i acc => (Tensor.Element.getPosition arr i).map (fun i => i :: acc)) [] 20
    return rev.reverse
  let _ <- debugPrintln actual
  match actual with
  | .error _ => return false
  | .ok actual => return expected == actual

-- TODO: Asserting true/false here would be great
private def iotrue : IO Bool := return true

#eval testTensorElementBV 16 "uint16" == iotrue
#eval testTensorElementBV 32 "uint16" -- expect false
#eval testTensorElementBV 64 "uint16" -- expect false
#eval testTensorElementBV 16 "uint32" -- expect false
#eval testTensorElementBV 32 "uint32" -- expect true
#eval testTensorElementBV 64 "uint32" -- expect false
#eval testTensorElementBV 16 "uint64" -- expect false
#eval testTensorElementBV 32 "uint64" -- expect false
#eval testTensorElementBV 64 "uint64" -- expect true

end Test
