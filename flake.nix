{
  description = "Jax devshell";

  nixConfig = {
    extra-substituters = [
      "https://nix-community.cachix.org"
    ];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    ];
  };

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }: let
    systems = ["x86_64-linux"];
  in
    flake-utils.lib.eachSystem systems (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        packages = [
          pkgs.python310
          pkgs.python310Packages.venvShellHook
          pkgs.uv
        ];

        libs = [
          pkgs.cudaPackages.cudatoolkit
          pkgs.cudaPackages.cudnn
          pkgs.stdenv.cc.cc.lib
          pkgs.zlib

          # Where your local "lib/libcuda.so" lives. If you're not on NixOS,
          # you should provide the right path (likely another one).
          "/run/opengl-driver"
        ];

        shell = pkgs.mkShell {
          name = "gnn-ranking";
          inherit packages;

          env = {
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath libs;
          };

          venvDir = "./.venv";
          postVenvCreation = ''
            uv sync
          '';
          postShellHook = ''
            python3 -c "import torch; print('CUDA available' if torch.cuda.is_available() else 'CPU only')"
            export SHELL="/run/current-system/sw/bin/bash"
          '';
        };
      in {
        devShells.default = shell;
      }
    );
}
