"""RoboTwin eval client for unified80 checkpoints.

Supports:
  - sparse 80-D action unpack → env 14-D ``[L6, Lg, R6, Rg]``
  - optional structured embodiment prompt (match pretrain / post-train)
  - optional proprio state pack + min_max normalize (80-D)
  - optional execution_horizon + chunked temporal ensemble
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Optional

import cv2 as cv
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


class ChunkedAdaptiveEnsembler:
    """Temporal ensemble for chunked policies (replan every N steps).

    Registers full predicted chunks and, at each env step, averages all
    still-valid predictions for that timestep (cosine-similarity weights).
    Overlap requires execution_horizon < model chunk length.
    """

    def __init__(self, max_chunks: int, adaptive_ensemble_alpha: float = 0.1):
        self.max_chunks = max(1, int(max_chunks))
        self.adaptive_ensemble_alpha = float(adaptive_ensemble_alpha)
        self.action_history: list[dict] = []
        self.current_step = 0

    def reset(self) -> None:
        self.action_history = []
        self.current_step = 0

    def ensemble_action(self, new_action_chunk: np.ndarray) -> None:
        self.action_history.append(
            {"start_step": self.current_step, "actions": np.asarray(new_action_chunk)}
        )
        if len(self.action_history) > self.max_chunks:
            self.action_history = self.action_history[-self.max_chunks :]

    def step(self) -> np.ndarray:
        self.action_history = [
            h
            for h in self.action_history
            if h["start_step"] + len(h["actions"]) > self.current_step
        ]
        relevant = []
        for item in self.action_history:
            idx = self.current_step - item["start_step"]
            if 0 <= idx < len(item["actions"]):
                relevant.append(item["actions"][idx])
        if not relevant:
            raise ValueError(
                f"Step {self.current_step}: no ensembled actions available "
                "(register a chunk before stepping)."
            )
        preds = np.stack(relevant, axis=0)
        if preds.shape[0] == 1:
            out = preds[0]
        else:
            ref = preds[-1]
            dots = np.sum(preds * ref, axis=1)
            norms = np.linalg.norm(preds, axis=1) * (np.linalg.norm(ref) + 1e-7)
            cos = dots / (norms + 1e-7)
            weights = np.exp(self.adaptive_ensemble_alpha * cos)
            weights = weights / weights.sum()
            out = np.sum(weights[:, None] * preds, axis=0)
        self.current_step += 1
        return out


def _load_format_embodiment_prompt():
    """Load formatter without importing ``starVLA.dataloader`` (needs accelerate)."""
    try:
        from starVLA.dataloader.gr00t_lerobot.embodiment_prompt import format_embodiment_prompt as _fn

        return _fn
    except Exception:
        pass
    try:
        import importlib.util

        prompt_path = (
            Path(__file__).resolve().parents[3]
            / "starVLA"
            / "dataloader"
            / "gr00t_lerobot"
            / "embodiment_prompt.py"
        )
        if not prompt_path.is_file():
            return None
        spec = importlib.util.spec_from_file_location("_starvla_embodiment_prompt", prompt_path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "format_embodiment_prompt", None)
    except Exception:
        return None


format_embodiment_prompt = _load_format_embodiment_prompt()


# ---------------------------------------------------------------------------
# ALOHA ↔ unified80 layout (see data-juicer aloha.yaml)
# Packed 14-D: [L6, Lg, R6, Rg]
# Unified arm block: 7 joints (slot1=shoulder_roll pad) + gripper @ +16
# ---------------------------------------------------------------------------
_UNIFIED_DIM = 80
_LEFT_BASE = 0
_RIGHT_BASE = 29
_OFF_GRIPPER = 16

# packed joint index → unified joint slot (None = pad / unused)
# packed: waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate
_PACKED_TO_SLOT = {
    0: 2,  # waist → shoulder_yaw
    1: 0,  # shoulder → shoulder_pitch
    2: 3,  # elbow → elbow_pitch
    3: 4,  # forearm_roll
    4: 5,  # wrist_angle → wrist_pitch
    5: 6,  # wrist_rotate → wrist_roll
}

# Env expects a reorder of the packed 14-D vector (legacy Robotwin interface).
_ENV_REORDER = np.array([0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13], dtype=np.int64)


def pack_aloha_14_to_unified80(vec14: np.ndarray) -> np.ndarray:
    """Map packed 14-D ALOHA state/action → sparse unified 80-D."""
    x = np.asarray(vec14, dtype=np.float32).reshape(-1)
    if x.shape[0] != 14:
        raise ValueError(f"Expected packed 14-D vector, got shape {x.shape}")
    out = np.zeros((_UNIFIED_DIM,), dtype=np.float32)
    for packed_i, slot in _PACKED_TO_SLOT.items():
        out[_LEFT_BASE + slot] = x[packed_i]
        out[_RIGHT_BASE + slot] = x[7 + packed_i]
    out[_LEFT_BASE + _OFF_GRIPPER] = x[6]
    out[_RIGHT_BASE + _OFF_GRIPPER] = x[13]
    return out


def unpack_unified80_to_aloha_14(vec80: np.ndarray) -> np.ndarray:
    """Map sparse unified 80-D → packed 14-D ALOHA."""
    x = np.asarray(vec80, dtype=np.float32).reshape(-1)
    if x.shape[0] != _UNIFIED_DIM:
        raise ValueError(f"Expected unified {_UNIFIED_DIM}-D vector, got shape {x.shape}")
    out = np.zeros((14,), dtype=np.float32)
    for packed_i, slot in _PACKED_TO_SLOT.items():
        out[packed_i] = x[_LEFT_BASE + slot]
        out[7 + packed_i] = x[_RIGHT_BASE + slot]
    out[6] = x[_LEFT_BASE + _OFF_GRIPPER]
    out[13] = x[_RIGHT_BASE + _OFF_GRIPPER]
    return out


def _minmax_normalize(x: np.ndarray, vmin: np.ndarray, vmax: np.ndarray) -> np.ndarray:
    """Training-time min_max: 2 * (x - min) / (max - min) - 1; inactive → 0."""
    x = np.asarray(x, dtype=np.float32)
    vmin = np.asarray(vmin, dtype=np.float32)
    vmax = np.asarray(vmax, dtype=np.float32)
    out = np.zeros_like(x, dtype=np.float32)
    mask = vmin != vmax
    out[mask] = 2.0 * (x[mask] - vmin[mask]) / (vmax[mask] - vmin[mask]) - 1.0
    return out


def _load_state_minmax_stats(ckpt_path: str, unnorm_key: str) -> tuple[np.ndarray, np.ndarray]:
    """Load state min/max (80-D) from checkpoint sidecar dataset_statistics.json."""
    ckpt = Path(ckpt_path)
    candidates = [
        ckpt.parent.parent / "dataset_statistics.json",  # .../run/checkpoints/steps_x.pt
        ckpt.parent / "dataset_statistics.json",
    ]
    stats_path = next((p for p in candidates if p.is_file()), None)
    if stats_path is None:
        raise FileNotFoundError(
            f"dataset_statistics.json not found near checkpoint {ckpt_path} "
            f"(looked in {[str(p) for p in candidates]})"
        )
    stats = json.loads(stats_path.read_text())
    if unnorm_key not in stats:
        raise KeyError(f"unnorm_key={unnorm_key!r} not in {list(stats.keys())} ({stats_path})")
    state = stats[unnorm_key]["state"]
    return np.asarray(state["min"], dtype=np.float32), np.asarray(state["max"], dtype=np.float32)


class ModelClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "robotwin",
        horizon: int = 0,
        action_ensemble=False,
        action_ensemble_horizon: Optional[int] = 3,
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha=0.1,
        host="127.0.0.1",
        port=5694,
        action_mode: str = "abs",
        normalization_mode: str = "min_max",
        # unified80 / embodiment-prompt options
        action_layout: str = "auto",  # auto | packed14 | unified80_aloha
        use_embodiment_prompt: bool = False,
        embodiment: str = "aloha",
        camera_view_direction: str = "opposite side",
        fps: int = 15,
        speed: Optional[int] = 500,
        include_state: bool = True,
        execution_horizon: Optional[int] = None,
    ) -> None:

        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        self.policy_ckpt_path = policy_ckpt_path

        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon
        self.action_ensemble = bool(action_ensemble)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = int(action_ensemble_horizon or 3)
        self.normalization_mode = normalization_mode

        self.action_mode = action_mode
        self.initial_state = None
        self.prev_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        self.num_image_history = 0

        self.action_chunk_size = None
        self.raw_actions = None

        server_meta = self.client.get_server_metadata()
        server_chunk = int(server_meta["action_chunk_size"])
        if execution_horizon is not None:
            self.action_chunk_size = min(server_chunk, int(execution_horizon))
        else:
            self.action_chunk_size = server_chunk
        self._server_chunk_size = server_chunk

        if self.action_ensemble:
            self.action_ensembler = ChunkedAdaptiveEnsembler(
                max_chunks=self.action_ensemble_horizon,
                adaptive_ensemble_alpha=self.adaptive_ensemble_alpha,
            )
            if self.action_chunk_size >= server_chunk:
                print(
                    "[WARN] action_ensemble=True but execution_horizon >= model chunk; "
                    "chunks do not overlap, so ensemble will not smooth replan seams. "
                    "Try execution_horizon < action_chunk_size (e.g. 30 when model=50)."
                )
        else:
            self.action_ensembler = None

        if self.unnorm_key is None:
            self.unnorm_key = server_meta.get("default_unnorm_key") or "new_embodiment"

        # Resolve action layout
        train_mix = server_meta.get("training_data_mix") or ""
        if action_layout == "auto":
            if "stack_bowls" in str(train_mix) or "unified80" in str(train_mix):
                action_layout = "unified80_aloha"
            elif server_meta.get("action_keys") == ["action.unified"]:
                action_layout = "unified80_aloha"
            else:
                action_layout = "packed14"
        self.action_layout = action_layout

        self.use_embodiment_prompt = bool(use_embodiment_prompt) and format_embodiment_prompt is not None
        if bool(use_embodiment_prompt) and format_embodiment_prompt is None:
            print(
                "[WARN] use_embodiment_prompt=True but format_embodiment_prompt could not be loaded; "
                "falling back to plain instruction text."
            )
        self.prompt_fields = {
            "embodiment": embodiment,
            "camera_view_direction": camera_view_direction,
            "fps": fps,
            "speed": speed,
        }
        self.include_state = bool(include_state) and self.action_layout == "unified80_aloha"

        self._state_min = self._state_max = None
        if self.include_state:
            self._state_min, self._state_max = _load_state_minmax_stats(
                policy_ckpt_path, self.unnorm_key
            )

        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_mode: {action_mode}, layout: {self.action_layout}, "
            f"embodiment_prompt: {self.use_embodiment_prompt}, "
            f"include_state: {self.include_state}, "
            f"execution_horizon: {self.action_chunk_size}/{server_chunk}, "
            f"action_ensemble: {self.action_ensemble} "
            f"(horizon={self.action_ensemble_horizon}, alpha={self.adaptive_ensemble_alpha}), "
            f"server_meta: {server_meta} ***"
        )

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.raw_actions = None
        self.initial_state = None
        self.prev_action = None

    def _build_lang(self, instruction: str) -> str:
        if not self.use_embodiment_prompt:
            return str(instruction)
        return format_embodiment_prompt(
            self.prompt_fields,
            str(instruction),
            field_dropout=False,
        )

    def _prepare_state(self, state14: np.ndarray) -> np.ndarray:
        """Pack + min_max-normalize env state to training-time unified80."""
        packed = pack_aloha_14_to_unified80(state14)
        return _minmax_normalize(packed, self._state_min, self._state_max)

    def step(
        self,
        example: dict,
        step: int = 0,
    ) -> np.ndarray:
        state = example.get("state", None)

        if self.action_mode in ["delta", "rel"] and self.initial_state is None:
            if state is None:
                raise ValueError(f"action_mode='{self.action_mode}' requires state to be provided in example")
            self.initial_state = np.array(state).copy()

        task_description = example.get("lang", None)
        images = example["image"]

        if example is not None:
            if task_description != self.task_description:
                self.reset(task_description)
                if self.action_mode in ["delta", "rel"] and state is not None:
                    self.initial_state = np.array(state).copy()

        images = [self._resize_image(image) for image in images]
        example["image"] = images
        example_copy = example.copy()

        # Build model-facing example
        example_copy["lang"] = self._build_lang(example_copy.get("lang", ""))
        if self.include_state and state is not None:
            norm_state = self._prepare_state(np.asarray(state, dtype=np.float32))
            example_copy["state"] = norm_state.reshape(1, -1)
        else:
            example_copy.pop("state", None)

        vla_input = {
            "examples": [example_copy],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }
        vla_input["unnorm_key"] = self.unnorm_key

        action_chunk_size = self.action_chunk_size

        if self.action_ensemble:
            need_replan = (step % action_chunk_size == 0) or (
                len(self.action_ensembler.action_history) == 0
            )
        else:
            need_replan = step % action_chunk_size == 0 or self.raw_actions is None

        if need_replan:
            # === TRAIN/TEST CONSISTENCY ===
            # unified80_aloha: state is 80-D normalized; lang may be embodiment prompt;
            # model returns 80-D unnormalized actions → unpack to packed 14-D.
            response = self.client.predict_action(vla_input)
            raw_actions = np.array(response["data"]["actions"][0])  # (chunk, D)

            if self.action_layout == "unified80_aloha":
                if raw_actions.shape[-1] != _UNIFIED_DIM:
                    raise ValueError(
                        f"unified80_aloha expects action_dim={_UNIFIED_DIM}, "
                        f"got {raw_actions.shape[-1]}"
                    )
                raw_actions = np.stack(
                    [unpack_unified80_to_aloha_14(a) for a in raw_actions],
                    axis=0,
                )

            if self.action_mode == "delta":
                raw_actions = self._delta_to_absolute(raw_actions, state)
            elif self.action_mode == "rel":
                raw_actions = self._rel_to_absolute(raw_actions)

            if self.action_ensemble:
                # Keep full model chunk so leftover steps can overlap the next replan.
                self.action_ensembler.ensemble_action(raw_actions)
                self.raw_actions = raw_actions
            else:
                # Client-side truncate: model may still return full server chunk.
                self.raw_actions = raw_actions[:action_chunk_size]

        if self.action_ensemble:
            current_action = self.action_ensembler.step()
        else:
            action_idx = step % action_chunk_size
            current_action = self.raw_actions[action_idx]

        if self.action_mode == "delta":
            self.prev_action = current_action.copy()

        # Legacy packed14 concat order is [L6, R6, Lg, Rg]; env wants [L6, Lg, R6, Rg].
        # unified80 unpack already returns env order — do not reorder again.
        if self.action_layout != "unified80_aloha":
            current_action = current_action[_ENV_REORDER]
        return current_action

    def _delta_to_absolute(self, delta_actions: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        abs_actions = np.zeros_like(delta_actions)
        base = self.prev_action if self.prev_action is not None else self.initial_state
        for i in range(len(delta_actions)):
            abs_actions[i] = delta_actions[i] + base
            base = abs_actions[i]
        return abs_actions

    def _rel_to_absolute(self, rel_actions: np.ndarray) -> np.ndarray:
        return rel_actions + self.initial_state

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image


def get_model(usr_args):
    policy_ckpt_path = usr_args.get("policy_ckpt_path")
    host = usr_args.get("host", "127.0.0.1")
    port = usr_args.get("port", 5694)
    unnorm_key = usr_args.get("unnorm_key", None)
    action_mode = usr_args.get("action_mode", "abs")
    normalization_mode = usr_args.get(
        "action_normalization_mode",
        usr_args.get("normalization_mode", "min_max"),
    )

    if policy_ckpt_path is None:
        raise ValueError("policy_ckpt_path must be provided in config")

    return ModelClient(
        policy_ckpt_path=policy_ckpt_path,
        host=host,
        port=port,
        unnorm_key=unnorm_key,
        action_mode=action_mode,
        normalization_mode=normalization_mode,
        action_layout=usr_args.get("action_layout", "auto"),
        use_embodiment_prompt=usr_args.get("use_embodiment_prompt", True),
        embodiment=usr_args.get("embodiment", "aloha"),
        camera_view_direction=usr_args.get("camera_view_direction", "opposite side"),
        fps=int(usr_args.get("fps", 15)),
        speed=usr_args.get("speed", 500),
        include_state=usr_args.get("include_state", True),
        execution_horizon=usr_args.get("execution_horizon", None),
        action_ensemble=bool(usr_args.get("action_ensemble", False)),
        action_ensemble_horizon=usr_args.get("action_ensemble_horizon", 3),
        adaptive_ensemble_alpha=float(usr_args.get("adaptive_ensemble_alpha", 0.1)),
    )


def reset_model(model):
    model.reset(task_description="")


def eval(TASK_ENV, model, observation):
    instruction = TASK_ENV.get_instruction()

    head_img = observation["observation"]["head_camera"]["rgb"]
    left_img = observation["observation"]["left_camera"]["rgb"]
    right_img = observation["observation"]["right_camera"]["rgb"]

    # Order: [head, left, right] to match training order (cam_high, left_wrist, right_wrist)
    images = [head_img, left_img, right_img]

    state = observation["joint_action"]["vector"]
    example = {
        "lang": str(instruction),
        "image": images,
        "state": state,
    }

    action = model.step(example, step=TASK_ENV.take_action_cnt)

    TASK_ENV.take_action(action)
