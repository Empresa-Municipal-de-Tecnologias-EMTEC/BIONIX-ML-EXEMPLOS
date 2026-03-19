#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
packages=(ativacoes autograd camadas computacao conjuntos dados graficos nucleo perdas uteis)
for d in "${packages[@]}"; do
  if [ -e "$d" ]; then
    echo "exists: $d"
  else
    target="/mnt/c/PROJETOS/BIONIX/BIONIX-ML/src/$d"
    if [ -e "$target" ]; then
      ln -s "$target" "$d"
      echo "linked: $d -> $target"
    else
      echo "missing target: $target"
    fi
  fi
done
