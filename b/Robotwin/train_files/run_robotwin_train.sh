#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root (b/Robotwin/train_files)

# This host has no bond0/InfiniBand; use the available Ethernet NIC and disable IB.
# Override NCCL_SOCKET_IFNAME / set NCCL_IB_DISABLE=0 on clusters that do have IB.
# Default ens3 (GCP); was enp0s19 on another host — wrong IFNAME → NCCL "no socket interface found".
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ens3}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1000}

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenGR00T
freeze_module_list=''
# Must contain "Qwen3.5"; needs transformers>=5.2
base_vlm=/home/luogang/share/zwy/CKPT/Qwen3.5-0.8B
config_yaml=./b/Robotwin/train_files/starvla_cotrain_robotwin_abs.yaml
# unified80 Qwen3.5-GR00T pretrain (dir → latest steps_*_pytorch_model.pt)
# pretrained_checkpoint=${PRETRAINED_CHECKPOINT:-/home/luogang/share/zwy/CKPT/0717_pad80_pretrain_qwen35_gr00t/checkpoints/steps_20000_pytorch_model.pt}
run_root_dir=./results/Checkpoints
data_mix=robotwin_stack_bowls_three_50
run_id=0721_${data_mix}_qwen35_08b
num_processes=${NUM_PROCESSES:-2}
# === End of environment variable configuration ===
###########################################################################################

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
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.include_state true \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps 20000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_Robotwin \
  --wandb_entity your_wandb_entity
  # --is_debug True


##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --trainer.pretrained_checkpoint ${pretrained_checkpoint} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
