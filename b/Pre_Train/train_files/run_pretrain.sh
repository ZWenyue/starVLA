#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root (b/Pre_Train/train_files)

# This host has no bond0/InfiniBand; use the available Ethernet NIC and disable IB.
# Override NCCL_SOCKET_IFNAME / set NCCL_IB_DISABLE=0 on clusters that do have IB.
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp0s19}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1000}

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenGR00T
freeze_module_list=''
# Must contain "Qwen3.5"; needs transformers>=5.2
base_vlm=playground/Pretrained_models/Qwen3.5-4B
# Registry + canonical yaml live under examples/ (auto-discovered)
config_yaml=./examples/realRobots/RoboCOIN/train_files/starvla_cotrain_robocoin_abs.yaml
data_root_dir=/mnt/r/DATA/pre_train_v1/unified_80_new
data_mix=unified80_pretrain
run_root_dir=./results/Checkpoints
run_id=0719_${data_mix}_qwen35_gr00t_fixed
num_processes=${NUM_PROCESSES:-8}
# === End of environment variable configuration ===
###########################################################################################

TEMPLATE_DIR=./examples/realRobots/RoboCOIN/train_files/modality_templates
declare -A MODALITY_MAP=(
  [Cobot_Magic_clean_up_the_tableware]=cobot_cam_front.json
  [Cobot_Magic_pot_storage_steamer]=cobot_cam_front.json
  [Cobot_Magic_move_plate]=cobot_cam_high.json
  [Handle_Plates_20250619_001]=galaxea_r1_lite.json
  [Organize_Refrigerator_Items0250703_002]=galaxea_r1_lite.json
)
for ds in "${!MODALITY_MAP[@]}"; do
  src="${TEMPLATE_DIR}/${MODALITY_MAP[$ds]}"
  dst="${data_root_dir}/${ds}/meta/modality.json"
  if [[ ! -f "${dst}" ]] || ! cmp -s "${src}" "${dst}"; then
    echo "[info] Installing modality.json -> ${dst}"
    cp "${src}" "${dst}"
  fi
done

# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"
cp "${config_yaml}" "${output_dir}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --datasets.vla_data.data_root_dir "${data_root_dir}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.include_state true \
  --datasets.vla_data.include_action_mask true \
  --datasets.vla_data.use_embodiment_prompt true \
  --datasets.vla_data.embodiment_prompt_field_dropout true \
  --datasets.vla_data.embodiment_prompt_field_dropout_prob 0.15 \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps 20000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_unified80 \
  --wandb_entity your_wandb_entity
  # --is_debug True
