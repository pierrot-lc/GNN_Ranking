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
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {inherit system;};
    packages = [
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
      name = "GNN_Ranking-repro";
      inherit packages;
      env = {
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath libs;
        XLA_FLAGS = "--xla_gpu_cuda_data_dir=${pkgs.cudatoolkit}";
      };
    };
  in {
    devShells.default = shell;
  };
}
