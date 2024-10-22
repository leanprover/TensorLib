import Lake
open Lake DSL

package "TensorLib" where
  -- add package configuration options here

lean_lib «TensorLib» where
  -- add library configuration options here

require scilean from git
  "https://github.com/lecopivo/SciLean.git"

@[default_target]
lean_exe "tensorlib" where
  root := `Main
