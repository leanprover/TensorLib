import Lake
open Lake DSL

package "TensorLib" where
  -- add package configuration options here

lean_lib «TensorLib» where
  -- add library configuration options here

@[default_target]
lean_exe "tensorlib" where
  root := `Main

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.12.0"

require aesop from git
  "https://github.com/leanprover-community/aesop" @ "v4.12.0"

require Cli from git
  "https://github.com/leanprover/lean4-cli.git" @ "v2.2.0-lv4.9.0"
