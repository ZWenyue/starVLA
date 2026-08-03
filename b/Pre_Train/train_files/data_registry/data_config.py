"""unified_80 pretrain — multi-embodiment datasets under /mnt/r/DATA/pre_train_v1/unified_80.

All tasks share the canonical 80-D state/action layout. Cameras are normalized to
``video.view0 / view1 / view2`` via per-dataset ``meta/modality.json``.
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


class Unified80DataConfig:
    """Full 80-D unified state/action + 3 canonical camera views."""

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.view0", "video.view1", "video.view2"]
    state_keys = ["state.unified"]
    action_keys = ["action.unified"]
    action_key_dims = {"action.unified": 80}
    state_key_dims = {"state.unified": 80}
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(50))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={"state.unified": "min_max"},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={"action.unified": "min_max"},
                ),
            ]
        )


class Unified80H16DataConfig(Unified80DataConfig):
    action_indices = list(range(16))


ROBOT_TYPE_CONFIG_MAP = {
    "unified80": Unified80H16DataConfig(),
    "unified80_50": Unified80DataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {}

DATASET_NAMED_MIXTURES = {
    "unified80_pretrain": [
        ("Cobot_Magic_clean_up_the_tableware", 1.0, "unified80_50"),
        ("Cobot_Magic_move_plate", 1.0, "unified80_50"),
        ("Cobot_Magic_pot_storage_steamer", 1.0, "unified80_50"),
        ("Handle_Plates_20250619_001", 1.0, "unified80_50"),
        ("Organize_Refrigerator_Items0250703_002", 1.0, "unified80_50"),
    ],
    "unified80_pretrain_h16": [
        ("Cobot_Magic_clean_up_the_tableware", 1.0, "unified80"),
        ("Cobot_Magic_move_plate", 1.0, "unified80"),
        ("Cobot_Magic_pot_storage_steamer", 1.0, "unified80"),
        ("Handle_Plates_20250619_001", 1.0, "unified80"),
        ("Organize_Refrigerator_Items0250703_002", 1.0, "unified80"),
    ],
    "unified80_cobot_magic": [
        ("Cobot_Magic_clean_up_the_tableware", 1.0, "unified80_50"),
        ("Cobot_Magic_move_plate", 1.0, "unified80_50"),
        ("Cobot_Magic_pot_storage_steamer", 1.0, "unified80_50"),
    ],
    "unified80_galaxea_r1_lite": [
        ("Handle_Plates_20250619_001", 1.0, "unified80_50"),
        ("Organize_Refrigerator_Items0250703_002", 1.0, "unified80_50"),
    ],
    # Robotwin task converted to unified80 layout (post-train)
    "stack_bowls_three": [
        ("stack_bowls_three", 1.0, "unified80_50"),
    ],
}
