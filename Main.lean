/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/
import TensorLib
import TensorLib.NumpyRepr
import Cli

open Cli
open TensorLib

def parseNpy (p: Parsed) : IO UInt32 := do
  let file := p.positionalArg! "input" |>.as! String
  IO.println s!"Parsing {file}..."
  let v <- NumpyRepr.Parse.parseFile file
  match v with
  | .error msg => IO.println s!"Couldn't parse {file}: {msg}"
  | .ok r => IO.println (repr r)
  return 0

def parseNpyCmd := `[Cli|
  "parse-npy" VIA parseNpy;
  "Parse a .npy file and pretty print the contents"

  ARGS:
    input : String;      ".npy file to parse"
]

def tensorlibCmd : Cmd := `[Cli|
  tensorlib NOOP; ["0.0.1"]
  "TensorLib is a NumPy-like library for Lean."

  SUBCOMMANDS:
    parseNpyCmd
]

def main (args : List String) : IO UInt32 :=
  tensorlibCmd.validate args
