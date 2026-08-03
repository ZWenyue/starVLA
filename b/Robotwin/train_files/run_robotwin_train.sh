#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root (b/Robotwin/train_files)

# This host has no bond0/InfiniBand; use the available Ethernet NIC and disable IB.
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
config_yaml=./b/Robotwin/train_files/starvla_cotrain_robotwin_abs.yaml
# unified80-converted Robotwin post-train data (80-D + action_dim_mask)
data_root_dir=/mnt/r/DATA/pre_train_v1/process_pad_unified80_new
data_mix=stack_bowls_three
# unified80 Qwen3.5-GR00T pretrain (dir → latest steps_*_pytorch_model.pt)
pretrained_checkpoint=${PRETRAINED_CHECKPOINT:-./results/Checkpoints/0719_unified80_pretrain_qwen35_gr00t_fixed/checkpoints}
run_root_dir=./results/Checkpoints
run_id=0721_${data_mix}gcl
num_processes=${NUM_PROCESSES:-8}
# === End of environment variable configuration ===
###########################################################################################

# Install unified80 modality.json (camera keys are cam_* without _rgb suffix)
TEMPLATE_DIR=./b/Robotwin/train_files/modality_templates
declare -A MODALITY_MAP=(
  [stack_bowls_three]=robotwin_cam.json
)
for ds in "${!MODALITY_MAP[@]}"; do
  src="${TEMPLATE_DIR}/${MODALITY_MAP[$ds]}"
  dst="${data_root_dir}/${ds}/meta/modality.json"
  if [[ ! -f "${dst}" ]] || ! cmp -s "${src}" "${dst}"; then
    echo "[info] Installing modality.json -> ${dst}"
    mkdir -p "$(dirname "${dst}")"
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
  --framework.qwenvl.vl_hidden_dim 2560 \
  --framework.action_model.action_dim 80 \
  --framework.action_model.state_dim 80 \
  --datasets.vla_data.data_root_dir "${data_root_dir}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.include_state true \
  --datasets.vla_data.include_action_mask true \
  --datasets.vla_data.use_embodiment_prompt true \
  --datasets.vla_data.embodiment_prompt_field_dropout true \
  --datasets.vla_data.embodiment_prompt_field_dropout_prob 0.15 \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.pretrained_checkpoint "${pretrained_checkpoint}" \
  --trainer.max_train_steps 20000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_Robotwin \
  --wandb_entity your_wandb_entity
  # --is_debug True
