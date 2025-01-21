{
  description = "Jax devshell";

  nixConfig = {
    extra-substituters = [
      "https://cuda-maintainers.cachix.org"
      "https://ploop.cachix.org"
    ];
    extra-trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
      "ploop.cachix.org-1:i6+Fqarsbf5swqH09RXOEDvxy7Wm7vbiIXu4A9HCg1g="
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
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        python-packages = ps:
          with ps; [
            pip
            setuptools
            virtualenv
          ];

        packages = [
          (pkgs.python310.withPackages python-packages)
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

          shellHook = ''
            if [ -d ".venv" ]; then
              source .venv/bin/activate
            fi

            export SHELL="/run/current-system/sw/bin/bash"
          '';

          env = {
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath libs;
          };
        };
      in {
        devShells.default = shell;
      }
    );
}
