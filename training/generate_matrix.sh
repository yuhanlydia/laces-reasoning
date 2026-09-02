#!/usr/bin/env bash
set -euo pipefail

TRAINING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
contexts=(512 4096)
backbones=(0.4B 2.9B 13.3B)
bases=(8 16 32 64)
objectives=(ddpm rf flow)
arches=(dit rwkv birwkv)

write_single() {
  local context="$1" backbone="$2" basis="$3" objective="$4"
  local dir="${TRAINING_DIR}/unconditional/${context}/${backbone}/single_z/basis_${basis}"
  mkdir -p "${dir}"
  cat > "${dir}/${objective}.sh" <<EOF2
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_LENGTH=${context} BACKBONE=${backbone} BASIS=${basis} OBJECTIVE=${objective} exec "\${SCRIPT_DIR}/../../../../../run_single_z.sh" "\$@"
EOF2
  chmod +x "${dir}/${objective}.sh"
}

write_traj() {
  local context="$1" backbone="$2" arch="$3" basis="$4" objective="$5"
  local dir="${TRAINING_DIR}/unconditional/${context}/${backbone}/trajectory/${arch}/basis_${basis}"
  mkdir -p "${dir}"
  cat > "${dir}/${objective}.sh" <<EOF2
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_LENGTH=${context} BACKBONE=${backbone} ARCH=${arch} BASIS=${basis} OBJECTIVE=${objective} exec "\${SCRIPT_DIR}/../../../../../../run_trajectory.sh" "\$@"
EOF2
  chmod +x "${dir}/${objective}.sh"
}

write_prefix_suffix_single() {
  local context="$1" backbone="$2" basis="$3" objective="$4"
  local dir="${TRAINING_DIR}/conditional/prefix_suffix/${context}/${backbone}/single_z/basis_${basis}"
  mkdir -p "${dir}"
  cat > "${dir}/${objective}.sh" <<EOF2
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_LENGTH=${context} BACKBONE=${backbone} BASIS=${basis} OBJECTIVE=${objective} exec "\${SCRIPT_DIR}/../../../../../../run_prefix_suffix_single_z.sh" "\$@"
EOF2
  chmod +x "${dir}/${objective}.sh"
}

write_prefix_suffix_traj() {
  local context="$1" backbone="$2" arch="$3" basis="$4" objective="$5"
  local dir="${TRAINING_DIR}/conditional/prefix_suffix/${context}/${backbone}/trajectory/${arch}/basis_${basis}"
  mkdir -p "${dir}"
  cat > "${dir}/${objective}.sh" <<EOF2
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_LENGTH=${context} BACKBONE=${backbone} ARCH=${arch} BASIS=${basis} OBJECTIVE=${objective} exec "\${SCRIPT_DIR}/../../../../../../../run_prefix_suffix_trajectory.sh" "\$@"
EOF2
  chmod +x "${dir}/${objective}.sh"
}

write_post_route() {
  local context="$1" backbone="$2" route="$3"
  local dir="${TRAINING_DIR}/conditional/prefix_suffix/${context}/${backbone}/post_training"
  mkdir -p "${dir}"
  cat > "${dir}/${route}.sh" <<EOF2
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_LENGTH=${context} BACKBONE=${backbone} ROUTE=${route} exec "\${SCRIPT_DIR}/../../../../../run_post_training_route.sh" "\$@"
EOF2
  chmod +x "${dir}/${route}.sh"
}

for context in "${contexts[@]}"; do
  for backbone in "${backbones[@]}"; do
    for basis in "${bases[@]}"; do
      for objective in "${objectives[@]}"; do
        write_single "${context}" "${backbone}" "${basis}" "${objective}"
      done
    done
    for arch in "${arches[@]}"; do
      for basis in "${bases[@]}"; do
        for objective in "${objectives[@]}"; do
          write_traj "${context}" "${backbone}" "${arch}" "${basis}" "${objective}"
        done
      done
    done
  done
done

for context in "${contexts[@]}"; do
  for backbone in "${backbones[@]}"; do
    for basis in "${bases[@]}"; do
      for objective in "${objectives[@]}"; do
        write_prefix_suffix_single "${context}" "${backbone}" "${basis}" "${objective}"
      done
    done
    for arch in "${arches[@]}"; do
      for basis in "${bases[@]}"; do
        for objective in "${objectives[@]}"; do
          write_prefix_suffix_traj "${context}" "${backbone}" "${arch}" "${basis}" "${objective}"
        done
      done
    done
    for route in planner_aware_s2 stochastic_continuation_s2 self_forcing_s2; do
      write_post_route "${context}" "${backbone}" "${route}"
    done
  done
done

mkdir -p "${TRAINING_DIR}/preprocess/4096"
for mode in full_rows pack_existing pack_stream; do
  cat > "${TRAINING_DIR}/preprocess/4096/${mode}.sh" <<EOF2
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
MODE=${mode} exec "\${SCRIPT_DIR}/../../run_preprocess_4096.sh" "\$@"
EOF2
  chmod +x "${TRAINING_DIR}/preprocess/4096/${mode}.sh"
done

echo "Generated training matrix under ${TRAINING_DIR}/{unconditional,conditional}/..."
