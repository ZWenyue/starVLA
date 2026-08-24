"""Generate architecture diagrams for evalother_2.md — InternVLA-A1.5 evaluation plan."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np


def draw_action_reorder_detail():
    """Visualize the action dimension reordering between InternVLA and starVLA."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    colors_L = '#42A5F5'  # blue for left arm
    colors_Lg = '#66BB6A'  # green for left gripper
    colors_R = '#EF5350'  # red for right arm
    colors_Rg = '#FFA726'  # orange for right gripper

    def draw_action_bar(ax, labels, colors, title, y_pos=0.5):
        ax.set_xlim(-0.5, 14.5)
        ax.set_ylim(0, 1.2)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.axis('off')

        for i, (label, color) in enumerate(zip(labels, colors)):
            rect = FancyBboxPatch((i, 0.2), 0.9, 0.7,
                                   boxstyle="round,pad=0.05",
                                   facecolor=color, edgecolor='#333', linewidth=1.5, alpha=0.85)
            ax.add_patch(rect)
            ax.text(i + 0.45, 0.55, label, ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white')
            ax.text(i + 0.45, 0.05, f'[{i}]', ha='center', va='center',
                    fontsize=7, color='#555')

    # Row 1: InternVLA env order
    labels1 = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'Lg', 'R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'Rg']
    colors1 = [colors_L]*6 + [colors_Lg] + [colors_R]*6 + [colors_Rg]
    draw_action_bar(axes[0], labels1, colors1,
                    'Step 0: InternVLA-A1.5 Output (env order)\n[L_joints(6), L_grip(1), R_joints(6), R_grip(1)]')

    # Row 2: starVLA model order (after server reorder)
    labels2 = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'Lg', 'Rg']
    colors2 = [colors_L]*6 + [colors_R]*6 + [colors_Lg] + [colors_Rg]
    draw_action_bar(axes[1], labels2, colors2,
                    'Step 1: Server Reorder → starVLA Model Order\n[L_joints(6), R_joints(6), L_grip(1), R_grip(1)]')

    # Row 3: RoboTwin env order (after client reorder)
    labels3 = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'Lg', 'R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'Rg']
    colors3 = [colors_L]*6 + [colors_Lg] + [colors_R]*6 + [colors_Rg]
    draw_action_bar(axes[2], labels3, colors3,
                    'Step 2: Client Reorder → RoboTwin Env Order\n[L_joints(6), L_grip(1), R_joints(6), R_grip(1)]')

    # Draw reorder arrows between rows using ConnectionPatch
    from matplotlib.patches import ConnectionPatch

    # Server reorder indices: [0,1,2,3,4,5,7,8,9,10,11,12,6,13]
    server_map = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]
    for dst_idx, src_idx in enumerate(server_map):
        if src_idx != dst_idx:
            con = ConnectionPatch(
                xyA=(dst_idx + 0.45, 0.85), coordsA=axes[1].transData,
                xyB=(src_idx + 0.45, 0.25), coordsB=axes[0].transData,
                arrowstyle='->', color='#E65100', lw=1.3, alpha=0.5)
            fig.add_artist(con)

    # Client reorder indices: [0,1,2,3,4,5,12,6,7,8,9,10,11,13]
    client_map = [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]
    for dst_idx, src_idx in enumerate(client_map):
        if src_idx != dst_idx:
            con = ConnectionPatch(
                xyA=(dst_idx + 0.45, 0.85), coordsA=axes[2].transData,
                xyB=(src_idx + 0.45, 0.25), coordsB=axes[1].transData,
                arrowstyle='->', color='#1565C0', lw=1.3, alpha=0.5)
            fig.add_artist(con)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors_L, label='Left Arm Joints (6D)'),
        mpatches.Patch(facecolor=colors_Lg, label='Left Gripper (1D)'),
        mpatches.Patch(facecolor=colors_R, label='Right Arm Joints (6D)'),
        mpatches.Patch(facecolor=colors_Rg, label='Right Gripper (1D)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=10, frameon=True, fancybox=True)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('b/d/rbt2/asset/action_reorder_detail.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved action_reorder_detail.png")


def draw_integration_architecture():
    """Draw the integration architecture showing dual-env setup."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 11))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis('off')
    ax.set_title('InternVLA-A1.5 + starVLA RoboTwin Eval: Integration Architecture',
                 fontsize=14, fontweight='bold', pad=20)

    # Styles
    new_style = dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4", edgecolor="#F9A825", linewidth=2)
    reuse_style = dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=1.5)
    env_box = dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=1.5)
    data_style = dict(boxstyle="round,pad=0.2", facecolor="#F3E5F5", edgecolor="#6A1B9A", linewidth=1.2)

    # Entry script
    ax.text(9, 10.2, 'run_internvla_a15_eval.sh', fontsize=11, ha='center', va='center',
            bbox=new_style, fontweight='bold')
    ax.text(9, 9.5, '(New: user entry, sets ROBOTWIN_SERVER_LAUNCH_SCRIPT)', fontsize=8,
            ha='center', va='center', color='#666')

    # start_eval.sh
    ax.text(9, 8.5, 'start_eval.sh (Reuse)\nParallel Task Scheduler\nGPU Slot Management',
            fontsize=9, ha='center', va='center', bbox=reuse_style)
    ax.annotate('', xy=(9, 8.9), xytext=(9, 9.7), arrowprops=dict(arrowstyle='->', lw=2, color='#333'))

    # conda env 1 (InternVLA)
    rect1 = plt.Rectangle((0.5, 3.5), 7, 4.5, fill=True, facecolor='#FFF8E1',
                           edgecolor='#F9A825', linewidth=2, linestyle='--', alpha=0.5)
    ax.add_patch(rect1)
    ax.text(4, 7.6, 'conda: internvla_a1_5', fontsize=10, ha='center', va='center',
            fontweight='bold', color='#E65100')

    # Server components
    ax.text(4, 6.8, 'run_server.sh (New)\nconda activate internvla_a1_5',
            fontsize=8, ha='center', va='center', bbox=new_style)
    ax.text(4, 5.5, 'internvla_a15_server.py (New)\nInternVLA15PolicyWrapper\n- from_pretrained(safetensors)\n- predict_action_chunk()\n- unnormalize (mean_std)\n- reorder → starVLA model order',
            fontsize=7.5, ha='center', va='center', bbox=new_style)
    ax.text(4, 3.8, 'WebsocketPolicyServer (Reuse)\nmsgpack-numpy, asyncio',
            fontsize=8, ha='center', va='center', bbox=reuse_style)

    ax.annotate('', xy=(4, 6.3), xytext=(4, 6.5), arrowprops=dict(arrowstyle='->', lw=1.5, color='#333'))
    ax.annotate('', xy=(4, 4.3), xytext=(4, 4.9), arrowprops=dict(arrowstyle='->', lw=1.5, color='#333'))

    # conda env 2 (RoboTwin)
    rect2 = plt.Rectangle((10, 3.5), 7.5, 4.5, fill=True, facecolor='#E8F5E9',
                           edgecolor='#2E7D32', linewidth=2, linestyle='--', alpha=0.3)
    ax.add_patch(rect2)
    ax.text(13.75, 7.6, 'conda: RoboTwin', fontsize=10, ha='center', va='center',
            fontweight='bold', color='#2E7D32')

    # Client components
    ax.text(13.75, 6.8, 'eval.sh (Reuse)\nPYTHONPATH setup',
            fontsize=8, ha='center', va='center', bbox=reuse_style)
    ax.text(13.75, 5.5, 'model2robotwin_interface.py (Reuse)\nModelClient\n- Resize images 224x224\n- WebSocket → predict_action\n- Chunk cache: actions[step % T]\n- Action reorder: model → env order',
            fontsize=7.5, ha='center', va='center', bbox=reuse_style)
    ax.text(13.75, 3.8, 'RoboTwin eval_policy.py\n50 tasks x 100 episodes\nSAPIEN Engine',
            fontsize=8, ha='center', va='center', bbox=env_box)

    ax.annotate('', xy=(13.75, 6.3), xytext=(13.75, 6.5), arrowprops=dict(arrowstyle='->', lw=1.5, color='#333'))
    ax.annotate('', xy=(13.75, 4.3), xytext=(13.75, 4.9), arrowprops=dict(arrowstyle='<->', lw=1.5, color='#333'))

    # start_eval to both envs
    ax.annotate('', xy=(4, 7.2), xytext=(6.5, 8.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='#E65100'))
    ax.annotate('', xy=(13.75, 7.2), xytext=(11.5, 8.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='#2E7D32'))

    # WebSocket connection
    ax.annotate('', xy=(10.2, 4.8), xytext=(7.3, 4.8),
                arrowprops=dict(arrowstyle='<->', color='#F44336', lw=2.5))
    ax.text(8.75, 4.8, 'WebSocket\nmsgpack-numpy', fontsize=8, ha='center', va='center',
            color='#F44336',
            bbox=dict(boxstyle="round,pad=0.15", facecolor='white', edgecolor='#F44336', linewidth=1.5))

    # Data flow labels
    ax.text(8.75, 5.6, '{images, lang}\n→ server', fontsize=7, ha='center', va='center',
            color='#6A1B9A', style='italic')
    ax.text(8.75, 4.0, '{actions [B,T,14]}\n← server', fontsize=7, ha='center', va='center',
            color='#6A1B9A', style='italic')

    # Legend at bottom
    ax.text(2, 2.8, 'Legend:', fontsize=9, fontweight='bold', va='center')
    leg_new = FancyBboxPatch((3.5, 2.55), 2, 0.5, boxstyle="round,pad=0.1",
                             facecolor="#FFF9C4", edgecolor="#F9A825", linewidth=1.5)
    ax.add_patch(leg_new)
    ax.text(4.5, 2.8, 'New', fontsize=8, ha='center', va='center')

    leg_reuse = FancyBboxPatch((6, 2.55), 2, 0.5, boxstyle="round,pad=0.1",
                                facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=1.5)
    ax.add_patch(leg_reuse)
    ax.text(7, 2.8, 'Reuse', fontsize=8, ha='center', va='center')

    leg_ext = FancyBboxPatch((8.5, 2.55), 2.5, 0.5, boxstyle="round,pad=0.1",
                              facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=1.5)
    ax.add_patch(leg_ext)
    ax.text(9.75, 2.8, 'Third-party', fontsize=8, ha='center', va='center')

    plt.tight_layout()
    plt.savefig('b/d/rbt2/asset/integration_architecture.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved integration_architecture.png")


def draw_model_comparison():
    """Draw a comparison between InternVLA-M1 and InternVLA-A1.5."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

    def draw_model_arch(ax, title, components, color_main, color_accent):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

        y_pos = 8.5
        for name, desc, is_main in components:
            color = color_main if is_main else color_accent
            style = dict(boxstyle="round,pad=0.3", facecolor=color,
                        edgecolor='#333', linewidth=1.5)
            ax.text(5, y_pos, f'{name}\n{desc}', fontsize=9, ha='center', va='center',
                    bbox=style)
            if y_pos > 1.5:
                ax.annotate('', xy=(5, y_pos - 0.5), xytext=(5, y_pos - 0.7),
                           arrowprops=dict(arrowstyle='->', lw=1.5, color='#333'))
            y_pos -= 1.8

    # InternVLA-M1
    m1_components = [
        ('Qwen2.5-VL / Qwen3-VL (4B)', 'VLM Backbone', True),
        ('DINOv2 (dinov2_vits14)', 'Multi-view Visual Encoder', True),
        ('Layer-wise QFormer', 'Layers 20-36, 64 queries\ninput_dim=2048, output_dim=768', True),
        ('DiT Diffusion Head (DiT-B)', 'DDIM Sampling (5 steps)\naction_dim=7, chunk=16', True),
        ('Output: normalized_actions', '[B, T, 7] (single arm)', False),
    ]
    draw_model_arch(ax1, 'InternVLA-M1 (in starVLA codebase)',
                    m1_components, '#BBDEFB', '#E3F2FD')

    # InternVLA-A1.5
    a15_components = [
        ('Qwen3.5-2B', 'VLM Backbone\n(with built-in vision encoder)', True),
        ('Unified Action Expert', 'Shared Full-Attention Layers\n+ Gated DeltaNet', True),
        ('Flow Matching Head', 'Continuous Normalizing Flow\nFAST Action Tokens', True),
        ('compact_reorder + unnorm', 'compact_reordered_dual_arm_actions\n+ mean_std unnormalize', True),
        ('Output: 14D absolute actions', '[T, 14] (dual arm)\n[L6, Lg, R6, Rg]', False),
    ]
    draw_model_arch(ax2, 'InternVLA-A1.5-base (external model)',
                    a15_components, '#FFE0B2', '#FFF3E0')

    # Cross-comparison annotations
    fig.text(0.5, 0.02,
             'INCOMPATIBLE: Different VLM backbones, different action heads, '
             'different checkpoint formats (PT vs safetensors), '
             'different normalization (min_max vs mean_std)',
             ha='center', fontsize=10, color='#C62828', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFCDD2',
                      edgecolor='#C62828', linewidth=1.5))

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig('b/d/rbt2/asset/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved model_comparison.png")


def draw_env_isolation():
    """Draw the dual conda environment isolation diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Dual Conda Environment Isolation',
                 fontsize=14, fontweight='bold', pad=20)

    # Env 1: internvla_a1_5
    rect1 = plt.Rectangle((0.5, 1), 6.5, 6, fill=True, facecolor='#FFF8E1',
                           edgecolor='#F9A825', linewidth=2.5, alpha=0.4)
    ax.add_patch(rect1)
    ax.text(3.75, 6.7, 'conda: internvla_a1_5', fontsize=12, ha='center',
            fontweight='bold', color='#E65100')

    deps1 = [
        'Python 3.11',
        'PyTorch 2.10.0 (CUDA 12.8)',
        'transformers 5.2.0 (patched)',
        'flash-attn 2.8.3',
        'flash-linear-attention 0.5.0',
        'causal-conv1d 1.6.1',
        'InternVLA-A-series (pip -e)',
        'websockets + msgpack-numpy',
    ]
    for i, dep in enumerate(deps1):
        ax.text(3.75, 6.0 - i * 0.55, dep, fontsize=8.5, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.15", facecolor='#FFF9C4',
                         edgecolor='#F9A825', linewidth=1))

    # Env 2: RoboTwin
    rect2 = plt.Rectangle((9, 1), 6.5, 6, fill=True, facecolor='#E8F5E9',
                           edgecolor='#2E7D32', linewidth=2.5, alpha=0.3)
    ax.add_patch(rect2)
    ax.text(12.25, 6.7, 'conda: RoboTwin', fontsize=12, ha='center',
            fontweight='bold', color='#2E7D32')

    deps2 = [
        'Python 3.x',
        'PyTorch (CUDA)',
        'sapien (SAPIEN physics)',
        'cv2 (OpenCV)',
        'numpy',
        'websockets + msgpack-numpy',
        'starVLA (PYTHONPATH only)',
        '(no model weights loaded)',
    ]
    for i, dep in enumerate(deps2):
        ax.text(12.25, 6.0 - i * 0.55, dep, fontsize=8.5, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.15", facecolor='#C8E6C9',
                         edgecolor='#2E7D32', linewidth=1))

    # WebSocket bridge
    ax.annotate('', xy=(9.2, 3.5), xytext=(6.8, 3.5),
                arrowprops=dict(arrowstyle='<->', color='#F44336', lw=3))
    ax.text(8, 3.5, 'WebSocket\nBridge', fontsize=10, ha='center', va='center',
            color='#F44336', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white',
                     edgecolor='#F44336', linewidth=2))

    ax.text(8, 0.5, 'No dependency conflicts — each environment runs independently',
            fontsize=10, ha='center', va='center', color='#333', style='italic')

    plt.tight_layout()
    plt.savefig('b/d/rbt2/asset/env_isolation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved env_isolation.png")


if __name__ == '__main__':
    draw_action_reorder_detail()
    draw_integration_architecture()
    draw_model_comparison()
    draw_env_isolation()
    print("\nAll diagrams generated successfully.")
