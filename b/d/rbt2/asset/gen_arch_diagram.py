"""Generate architecture diagrams for the starVLA RoboTwin evaluation pipeline analysis."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def draw_eval_dataflow():
    """Draw the complete evaluation data flow diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('starVLA RoboTwin Evaluation: End-to-End Data Flow', fontsize=14, fontweight='bold', pad=20)

    box_style = dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=1.5)
    server_style = dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0", edgecolor="#E65100", linewidth=1.5)
    env_style = dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=1.5)
    data_style = dict(boxstyle="round,pad=0.2", facecolor="#F3E5F5", edgecolor="#6A1B9A", linewidth=1.2)

    # RoboTwin Environment
    ax.text(2, 9, 'RoboTwin 2.0 Simulator\n(SAPIEN Engine)', fontsize=10, ha='center', va='center', bbox=env_style)
    # Observation
    ax.text(5.5, 9, 'Observation:\n- head_camera RGB\n- left_camera RGB\n- right_camera RGB\n- joint_action vector (14D)\n- instruction (str)', fontsize=8, ha='center', va='center', bbox=data_style)
    ax.annotate('', xy=(4, 9), xytext=(3.2, 9), arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5))

    # model2robotwin_interface
    ax.text(9, 9, 'model2robotwin_interface.py\nModelClient', fontsize=10, ha='center', va='center', bbox=box_style)
    ax.annotate('', xy=(7.5, 9), xytext=(7, 9), arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5))

    # Preprocessing
    ax.text(9, 7.5, 'Preprocessing:\n1. Resize images to 224x224\n2. Build lang instruction\n3. Pack {image, lang, state}\n4. Set unnorm_key', fontsize=8, ha='center', va='center', bbox=data_style)
    ax.annotate('', xy=(9, 8.4), xytext=(9, 8.6), arrowprops=dict(arrowstyle='->', color='#6A1B9A', lw=1.5))

    # WebSocket Client
    ax.text(9, 6, 'WebsocketClientPolicy\n(msgpack-numpy)', fontsize=9, ha='center', va='center', bbox=box_style)
    ax.annotate('', xy=(9, 6.6), xytext=(9, 6.9), arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5))

    # Network boundary
    ax.axhline(y=5.2, xmin=0.35, xmax=0.85, color='#F44336', linestyle='--', linewidth=2)
    ax.text(12, 5.2, 'WebSocket\n(msgpack)', fontsize=8, ha='center', va='center', color='#F44336',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='#F44336'))

    # Server side
    ax.text(9, 4.3, 'WebsocketPolicyServer\n_route_message() -> predict_action()', fontsize=9, ha='center', va='center', bbox=server_style)

    # PolicyServerWrapper
    ax.text(9, 3, 'PolicyServerWrapper\npredict_action(examples, unnorm_key)', fontsize=9, ha='center', va='center', bbox=server_style)
    ax.annotate('', xy=(9, 3.6), xytext=(9, 3.8), arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.5))

    # Two branches
    ax.text(5, 2, 'baseframework\n.predict_action()\n-> normalized_actions\n   [B, T, D]', fontsize=8, ha='center', va='center', bbox=server_style)
    ax.text(13, 2, 'PolicyNormProcessor\n.unapply_actions()\nnormalized -> env-space\n   [B, T, D]', fontsize=8, ha='center', va='center', bbox=server_style)
    ax.annotate('', xy=(6.5, 2.5), xytext=(7.5, 2.8), arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.5))
    ax.annotate('', xy=(11.5, 2.5), xytext=(10.5, 2.8), arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.5))
    ax.annotate('', xy=(11, 2), xytext=(7, 2), arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.5))

    # Return path
    ax.text(2, 1, 'Action (14D):\n[L_joints(6), L_grip(1),\n R_joints(6), R_grip(1)]', fontsize=8, ha='center', va='center', bbox=data_style)
    ax.text(2, 3, 'TASK_ENV\n.take_action(action)', fontsize=9, ha='center', va='center', bbox=env_style)
    ax.annotate('', xy=(2, 2), xytext=(2, 2.5), arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5))
    ax.annotate('', xy=(2, 3.8), xytext=(2, 8.3), arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5, connectionstyle="arc3,rad=0.3"))

    plt.tight_layout()
    plt.savefig('b/d/rbt2/asset/eval_dataflow.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved eval_dataflow.png")


def draw_coupling_diagram():
    """Draw the coupling points that block non-starVLA model evaluation."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Coupling Points Blocking Non-starVLA Model Evaluation', fontsize=14, fontweight='bold', pad=20)

    # Coupling points
    coupling_style = dict(boxstyle="round,pad=0.4", facecolor="#FFCDD2", edgecolor="#C62828", linewidth=2)
    ok_style = dict(boxstyle="round,pad=0.4", facecolor="#C8E6C9", edgecolor="#2E7D32", linewidth=2)
    neutral_style = dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=1.5)

    ax.text(7, 7.3, 'starVLA Evaluation Coupling Analysis', fontsize=12, ha='center', va='center', fontweight='bold')

    # CP1
    ax.text(3, 6, 'CP1: PolicyServerWrapper\nbaseframework.from_pretrained()\nRequires starVLA checkpoint format:\nconfig.yaml + dataset_statistics.json\n+ FRAMEWORK_REGISTRY', fontsize=8, ha='center', va='center', bbox=coupling_style)

    # CP2
    ax.text(10, 6, 'CP2: PolicyNormProcessor\n_resolve_robot_type() requires\ndata_mix in DATASET_NAMED_MIXTURES\n+ ROBOT_TYPE_CONFIG_MAP', fontsize=8, ha='center', va='center', bbox=coupling_style)

    # CP3
    ax.text(3, 3.5, 'CP3: predict_action() API\nReturns {"normalized_actions"}\nExpects unnorm_key for\ndenormalization stats lookup', fontsize=8, ha='center', va='center', bbox=coupling_style)

    # CP4
    ax.text(10, 3.5, 'CP4: Action Space Convention\n14D: [L6, R6, Lg, Rg] model order\nvs [L6, Lg, R6, Rg] env order\nReorder: [0,1,2,3,4,5,12,6,...,13]', fontsize=8, ha='center', va='center', bbox=coupling_style)

    # OK points
    ax.text(3, 1.2, 'OK: WebSocket Transport\nProtocol-agnostic msgpack\nAny client can connect', fontsize=8, ha='center', va='center', bbox=ok_style)

    ax.text(10, 1.2, 'OK: RoboTwin Interface\nDynamic import via policy_name\nCustom module can be used', fontsize=8, ha='center', va='center', bbox=ok_style)

    plt.tight_layout()
    plt.savefig('b/d/rbt2/asset/coupling_points.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved coupling_points.png")


def draw_adapter_strategy():
    """Draw the proposed adapter strategy for non-starVLA models."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Proposed Architecture: External Model Adapter for RoboTwin Eval', fontsize=13, fontweight='bold', pad=20)

    new_style = dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4", edgecolor="#F9A825", linewidth=2)
    existing_style = dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=1.5)
    env_style = dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=1.5)

    # Strategy A: Replace server
    ax.text(4, 9, 'Strategy A: Replace Policy Server', fontsize=11, ha='center', va='center', fontweight='bold')
    ax.text(4, 7.8, 'ExternalPolicyServer\n(NEW: implements same WebSocket API)\n- Loads any model natively\n- Handles its own normalization\n- Returns 14D unnormalized actions', fontsize=8, ha='center', va='center', bbox=new_style)
    ax.text(4, 6.2, 'model2robotwin_interface.py\n(UNCHANGED: connects via WebSocket)', fontsize=8, ha='center', va='center', bbox=existing_style)
    ax.text(4, 5, 'RoboTwin Simulator\n(UNCHANGED)', fontsize=8, ha='center', va='center', bbox=env_style)
    ax.annotate('', xy=(4, 7.1), xytext=(4, 6.7), arrowprops=dict(arrowstyle='->', color='#F9A825', lw=2))
    ax.annotate('', xy=(4, 5.6), xytext=(4, 5.8), arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5))

    # Strategy B: Replace interface
    ax.text(12, 9, 'Strategy B: Replace Client Interface', fontsize=11, ha='center', va='center', fontweight='bold')
    ax.text(12, 7.8, 'model2external_interface.py\n(NEW: loads model directly in-process)\n- No server needed\n- Implements get_model/eval/reset_model\n- Handles normalization internally', fontsize=8, ha='center', va='center', bbox=new_style)
    ax.text(12, 6.2, 'RoboTwin eval_policy.py\n(uses policy_name=\n "model2external_interface")', fontsize=8, ha='center', va='center', bbox=existing_style)
    ax.text(12, 5, 'RoboTwin Simulator\n(UNCHANGED)', fontsize=8, ha='center', va='center', bbox=env_style)
    ax.annotate('', xy=(12, 7.1), xytext=(12, 6.7), arrowprops=dict(arrowstyle='->', color='#F9A825', lw=2))
    ax.annotate('', xy=(12, 5.6), xytext=(12, 5.8), arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5))

    # Comparison table area
    ax.text(8, 3.5, 'Strategy Comparison', fontsize=11, ha='center', va='center', fontweight='bold')

    headers = ['Aspect', 'Strategy A (Replace Server)', 'Strategy B (Replace Client)']
    rows = [
        ['Minimal changes', 'New server only', 'New interface only'],
        ['Decoupling', 'Process-level isolation', 'In-process, simpler'],
        ['GPU sharing', 'Server owns GPU', 'Shares with RoboTwin'],
        ['Reuse existing code', 'High (WebSocket reuse)', 'Medium (new loader)'],
        ['Best for', 'Remote/distributed eval', 'Single-machine, quick test'],
    ]

    table = ax.table(cellText=rows, colLabels=headers, loc='center',
                     bbox=[0.05, 0.02, 0.9, 0.28])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#E3F2FD')
            cell.set_text_props(fontweight='bold')

    plt.tight_layout()
    plt.savefig('b/d/rbt2/asset/adapter_strategy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved adapter_strategy.png")


if __name__ == '__main__':
    draw_eval_dataflow()
    draw_coupling_diagram()
    draw_adapter_strategy()
