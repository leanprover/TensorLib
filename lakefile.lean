import Lake
open Lake DSL

package "TensorLib" where
  -- add package configuration options here

lean_lib «TensorLib» where
  -- add library configuration options here

@[default_target]
lean_exe "tensorlib" where
  root := `Main

require aesop from git
  "https://github.com/leanprover-community/aesop" @ "v4.18.0"

require plausible from git
  "https://github.com/leanprover-community/plausible" @ "v4.18.0"

require Cli from git
  "https://github.com/leanprover/lean4-cli.git" @ "v4.18.0"
