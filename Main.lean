/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/
import Cli
import Init.System.IO
import TensorLib

open Cli
open TensorLib

def parseNpy (p : Parsed) : IO UInt32 := do
  let file := p.positionalArg! "input" |>.as! String
  IO.println s!"Parsing {file}..."
  let v <- NumpyRepr.Parse.parseFile file
  match v with
  | .error msg =>
    IO.println s!"Couldn't parse {file}: {msg}"
    return 1
  | .ok r => do
    IO.println (repr r)
    if p.hasFlag "write" then do
      let new := (System.FilePath.mk file).addExtension "new"
      IO.println s!"Writing copy to {new}"
      let _ <- NumpyRepr.save! r new
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
  -- Just pytest for now, but add Lean tests here as well
  -- pytest will exit nonzero on it's own, so we don't need to check exit code
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
    parseNpyCmd;
    runTestsCmd
]

def main (args : List String) : IO UInt32 :=
  if args.isEmpty then do
    IO.println tensorlibCmd.help
    return 0
  else
    tensorlibCmd.validate args
