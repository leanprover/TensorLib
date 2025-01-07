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
  "https://github.com/leanprover-community/aesop" @ "master" -- 'master' rather than a tag is a workaround for segfault bug https://github.com/leanprover/lean4/issues/6518#issuecomment-2574607960

require Cli from git
  "https://github.com/leanprover/lean4-cli.git" @ "v2.2.0-lv4.14.0-rc1"
