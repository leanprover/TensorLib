/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/

import Init.System.IO
import Cli
import TensorLib
import TensorLib.Test

open Cli
open TensorLib

def format (p : Parsed) : IO UInt32 := do
  let shape : Shape := Shape.mk (p.variableArgsAs! Nat).toList
  IO.println s!"Got shape {shape}"
  let range := Tensor.arange! Dtype.uint16 shape.count
  let v := range.reshape! shape
  IO.println v.toNatTree!.format!
  return 0

def formatCmd := `[Cli|
  "format" VIA format;
  "Test formatting"

  ARGS:
    ...shape : Nat;      "shape to test"
]

def parseNpy (p : Parsed) : IO UInt32 := do
  let file := p.positionalArg! "input" |>.as! String
  IO.println s!"Parsing {file}..."
  let v <- Npy.parseFile file
  IO.println (repr v)
  if p.hasFlag "write" then do
    let new := (System.FilePath.mk file).addExtension "new"
    IO.println s!"Writing copy to {new}"
    let _ <- v.save! new
    -- TensorLib.Npy.save! (arr : Ndarray) (file : System.FilePath) : IO Unit
  return 0

def parseNpyCmd := `[Cli|
  "parse-npy" VIA parseNpy;
  "Parse a .npy file and pretty print the contents"

  FLAGS:
    write; "Also write the result back to `input`.new to test saving arrays to disk"

  ARGS:
    input : String;      ".npy file to parse"
]

def runTests (_ : Parsed) : IO UInt32 := do
  IO.println "Running Lean tests..."
  let t0 <- Test.runAllTests
  if !t0 then do
    IO.println "Lean tests failed"
    return 1
  IO.println "Running PyTest..."
  let output <- IO.Process.output { cmd := "pytest" }
  IO.println s!"stdout: {output.stdout}"
  IO.println s!"stderr: {output.stderr}"
  return output.exitCode

def runTestsCmd := `[Cli|
  "test" VIA runTests;
  "Run tests"
]

def tensorlibCmd : Cmd := `[Cli|
  tensorlib NOOP; ["0.0.1"]
  "TensorLib is a NumPy-like library for Lean."

  SUBCOMMANDS:
    formatCmd;
    parseNpyCmd;
    runTestsCmd
]

def main (args : List String) : IO UInt32 :=
  if args.isEmpty then do
    IO.println tensorlibCmd.help
    return 0
  else do
   tensorlibCmd.validate args
