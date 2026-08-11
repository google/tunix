# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RL Trainer."""

import hashlib
import json
import os
from typing import Any, Callable, Optional

from flax import nnx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
import optax
import numpy as np
from tunix.sft import peft_trainer
from typing_extensions import override
from tunix.perf import trace as perf_trace
from tunix.perf.experimental import tracer as perf_tracer_lib
from tunix.sft.metrics_logger import MetricsLogger  # pylint: disable=unused-import
from tunix.rl import alignment


class Trainer(peft_trainer.PeftTrainer):
  """Handles additional RL metrics logging and display."""

  supports_sequence_packing = True

  def __init__(
      self,
      model: nnx.Module,
      optimizer: optax.GradientTransformation,
      training_config: peft_trainer.TrainingConfig,
      custom_checkpoint_metadata_fn: Callable[[], dict[str, Any]],
      metrics_logger: Optional[MetricsLogger] = None,
      perf_tracer: Optional[perf_trace.Tracer] = None,
      perf_tracer_v2: Optional[perf_tracer_lib.Tracer] = None,
  ):
    super().__init__(
        model,
        optimizer,
        training_config,
        metrics_logger,
        perf_tracer,
        perf_tracer_v2,
    )
    self.rl_metrics_to_log = {}  # Metric name -> key in aux.
    self.tqdm_metrics_to_display = []
    self.custom_checkpoint_metadata_fn = custom_checkpoint_metadata_fn
    self._canon_alignment_sidecar = None
    self._canon_update_before = None

  @staticmethod
  def _canon_fingerprint_state(
      state: Any, *, max_leaves: int = 12, min_elements: int = 128
  ) -> dict:
    """Hash a deterministic, bounded sample of small floating state leaves."""
    if min_elements < 1:
      raise ValueError("min_elements must be positive")
    flat = jax.tree_util.tree_flatten_with_path(
        state, is_leaf=lambda value: isinstance(value, nnx.Variable)
    )[0]
    candidates = []
    for path, value in flat:
      array = value[...] if isinstance(value, nnx.Variable) else value
      shape = tuple(getattr(array, "shape", ()))
      dtype = getattr(array, "dtype", None)
      size = int(np.prod(shape, dtype=np.int64)) if shape else 1
      # NumPy does not classify ml_dtypes.bfloat16 as ``np.floating`` while
      # JAX does.  Reference models are intentionally stored in bf16, so use
      # JAX's dtype lattice for the state-immutability instrument.
      if dtype is None or not jnp.issubdtype(dtype, jnp.floating):
        continue
      if not min_elements <= size <= 1_048_576:
        continue
      candidates.append((jax.tree_util.keystr(path), array, shape, str(dtype)))
    if not candidates:
      raise alignment.AlignmentGateError("no bounded floating state leaves to hash")
    count = min(max_leaves, len(candidates))
    positions = np.linspace(0, len(candidates) - 1, count, dtype=np.int64)
    leaves = {}
    total_bytes = 0
    for position in positions.tolist():
      path, value, shape, dtype = candidates[position]
      host = np.ascontiguousarray(np.asarray(jax.device_get(value)))
      total_bytes += int(host.nbytes)
      leaves[path] = {
          "sha256": hashlib.sha256(host.tobytes()).hexdigest(),
          "shape": list(shape),
          "dtype": dtype,
          "bytes": int(host.nbytes),
      }
    return {
        "eligible_leaves": len(candidates),
        "sampled_leaves": len(leaves),
        "sampled_bytes": total_bytes,
        "leaves": leaves,
    }

  @staticmethod
  def _canon_changed_paths(before: dict, after: dict) -> list[str]:
    if set(before["leaves"]) != set(after["leaves"]):
      raise alignment.AlignmentGateError("update fingerprint leaf set changed")
    return [
        path
        for path in before["leaves"]
        if before["leaves"][path]["sha256"]
        != after["leaves"][path]["sha256"]
    ]

  @override
  def _prepare_inputs(self, input_data: Any) -> Any:
    if not alignment.enabled():
      return input_data
    mode = alignment.execution_mode()
    p27 = os.environ.get("CANON_FROZENLAKE_P27", "") == "1"
    if mode == "update-canary" or (mode == "gate-only" and p27):
      if self._canon_update_before is None:
        self._canon_update_before = {
            "model": self._canon_fingerprint_state(
                nnx.state(self.model, nnx.Param)
            ),
            "optimizer": self._canon_fingerprint_state(
                nnx.state(self.optimizer, nnx.optimizer.OptState)
            ),
            "train_steps": self.train_steps,
        }
        if p27:
          self._canon_update_before["accumulator"] = (
              self._canon_fingerprint_state(nnx.state(self.grad_accumulator))
          )
        print(
            "[CANON_FROZENLAKE_P27] pre_state_snapshot "
            if p27
            else "[CANON_GSM8K_UPDATE] pre_update_snapshot ",
            end="",
            flush=True,
        )
        print(
            f"model_leaves={self._canon_update_before['model']['sampled_leaves']} "
            f"optimizer_leaves={self._canon_update_before['optimizer']['sampled_leaves']} "
            f"train_steps={self.train_steps}",
            flush=True,
        )
    core, sidecar = alignment.unwrap_train_example(input_data)
    if sidecar is None:
      raise alignment.AlignmentGateError(
          "alignment gate enabled but train batch has no ObservedTrainExample sidecar"
      )
    if self._canon_alignment_sidecar is not None:
      raise alignment.AlignmentGateError(
          "previous alignment sidecar was not consumed; refusing batch reordering"
      )
    self._canon_alignment_sidecar = sidecar
    print("[CANON_ALIGN] host sidecar stripped before shard_input/JIT", flush=True)
    return core

  def with_rl_metrics_to_log(
      self,
      rl_metrics_to_log: dict[str, Callable[[ArrayLike], ArrayLike]],
  ) -> None:
    self.rl_metrics_to_log = rl_metrics_to_log

  def with_tqdm_metrics_to_display(
      self, tqdm_metrics_to_display: list[str | Callable[[], str]]
  ) -> None:
    self.tqdm_metrics_to_display = tqdm_metrics_to_display

  @override
  def custom_checkpoint_metadata(self) -> dict[str, Any]:
    return self.custom_checkpoint_metadata_fn()

  def restored_global_step(self) -> int:
    return self._restored_custom_metadata.get("global_step", 0)

  @override
  def _post_process_train_step(self, aux: Any) -> None:
    if alignment.enabled():
      sidecar = self._canon_alignment_sidecar
      self._canon_alignment_sidecar = None
      if sidecar is None:
        raise alignment.AlignmentGateError(
            "value_and_grad returned without a pending alignment sidecar"
        )
      required = (
          "canon/T_current",
          "canon/gradient_norm",
          "canon/optimizer_skipped",
          "canon/is_update_step",
      )
      missing = [key for key in required if key not in aux]
      if missing:
        raise alignment.AlignmentGateError(
            f"value_and_grad aux missing required alignment outputs: {missing}"
        )
      expected_red = os.environ.get("CANON_ALIGNMENT_EXPECTED_RED", "") == "1"
      if expected_red and (
          alignment.execution_mode() != "gate-only"
          or os.environ.get("CANON_FROZENLAKE_P27", "") != "1"
      ):
        raise alignment.AlignmentGateError(
            "CANON_ALIGNMENT_EXPECTED_RED is admitted only for P27 gate-only"
        )
      record = alignment.check_batch(
          sidecar,
          t_current=jax.device_get(aux["canon/T_current"]),
          gradient_norm=jax.device_get(aux["canon/gradient_norm"]),
          optimizer_skipped=jax.device_get(aux["canon/optimizer_skipped"]),
          step=self.train_steps,
          fail_closed=not expected_red,
      )
      if record["execution_mode"] == "train":
        if self._buffered_train_metrics is None:
          raise alignment.AlignmentGateError(
              "alignment record exists without a train metrics buffer"
          )
        boundaries = record["boundaries"]
        exact = record["exact"]
        warning_count = len(record.get("warning_reds", ()))
        scalars = {
            "zero_tim/hard_gate_pass": float(
                record["verdict"] == "PASS"
            ),
            "zero_tim/alignment_warning": float(warning_count > 0),
            "zero_tim/alignment_warning_count": float(warning_count),
            "zero_tim/n_action": float(record["N_action"]),
            "zero_tim/s_decode_vs_s_prefill_bytes": float(
                boundaries["S_decode_vs_S_prefill"]["differing_bytes"]
            ),
            "zero_tim/s_prefill_vs_t_old_bytes": float(
                boundaries["S_prefill_vs_T_old"]["differing_bytes"]
            ),
            "zero_tim/t_old_vs_t_current_bytes": float(
                boundaries["T_old_vs_T_current"]["differing_bytes"]
            ),
            "zero_tim/s_decode_vs_s_prefill_max_abs": float(
                boundaries["S_decode_vs_S_prefill"]["max_abs"]
            ),
            "zero_tim/s_prefill_vs_t_old_max_abs": float(
                boundaries["S_prefill_vs_T_old"]["max_abs"]
            ),
            "zero_tim/t_old_vs_t_current_max_abs": float(
                boundaries["T_old_vs_T_current"]["max_abs"]
            ),
            "zero_tim/w_exact": float(exact["w_all_exactly_1"]),
            "zero_tim/r_exact": float(exact["r_all_exactly_1"]),
            "zero_tim/wr_exact": float(exact["wr_all_exactly_1"]),
            "zero_tim/clip_hits": float(record["clip_hits"]),
            "zero_tim/tis_hits": float(record["tis_hits"]),
            "zero_tim/w_min": float(record["ratio_stats"]["w"]["min"]),
            "zero_tim/w_max": float(record["ratio_stats"]["w"]["max"]),
            "zero_tim/r_min": float(record["ratio_stats"]["r"]["min"]),
            "zero_tim/r_max": float(record["ratio_stats"]["r"]["max"]),
            "zero_tim/wr_min": float(record["ratio_stats"]["wr"]["min"]),
            "zero_tim/wr_max": float(record["ratio_stats"]["wr"]["max"]),
            "zero_tim/gradient_nonzero": float(record["gradient"]["nonzero"]),
            "zero_tim/first_order_kl": float(
                record["kl_protocol"]["first_order_-mean_delta"]
            ),
            "zero_tim/second_order_kl": float(
                record["kl_protocol"]["second_order_half_mean_delta2"]
            ),
        }
        for boundary_name, metric_prefix in (
            ("S_decode_vs_S_prefill", "s_decode_vs_s_prefill"),
            ("S_prefill_vs_T_old", "s_prefill_vs_t_old"),
            ("T_old_vs_T_current", "t_old_vs_t_current"),
        ):
          boundary = boundaries[boundary_name]
          scalars[f"zero_tim/{metric_prefix}_elements"] = float(
              boundary["differing_elements"]
          )
          scalars[f"zero_tim/{metric_prefix}_element_fraction"] = float(
              boundary["element_fraction"]
          )
          scalars[f"zero_tim/{metric_prefix}_byte_fraction"] = float(
              boundary["byte_fraction"]
          )
        for metric_name, value in scalars.items():
          entry = self._buffered_train_metrics.additional_metrics.get(
              metric_name
          )
          if entry is None:
            self._buffered_train_metrics.additional_metrics[metric_name] = (
                [value],
                np.mean,
            )
          else:
            entry[0].append(value)
      if (
          record["execution_mode"] == "gate-only"
          and os.environ.get("CANON_FROZENLAKE_P27", "") == "1"
      ):
        is_update_step = bool(
            np.asarray(jax.device_get(aux["canon/is_update_step"])).item()
        )
        before = self._canon_update_before
        if before is None:
          raise alignment.AlignmentGateError(
              "P27 gate-only state snapshot missing after train step"
          )
        if not is_update_step:
          print(
              "[CANON_FROZENLAKE_P27] gate_accumulation_pending "
              f"train_steps={self.train_steps}",
              flush=True,
          )
        else:
          self._canon_update_before = None
          after_model = self._canon_fingerprint_state(
              nnx.state(self.model, nnx.Param)
          )
          after_optimizer = self._canon_fingerprint_state(
              nnx.state(self.optimizer, nnx.optimizer.OptState)
          )
          after_accumulator = self._canon_fingerprint_state(
              nnx.state(self.grad_accumulator)
          )
          model_changed = self._canon_changed_paths(
              before["model"], after_model
          )
          optimizer_changed = self._canon_changed_paths(
              before["optimizer"], after_optimizer
          )
          accumulator_changed = self._canon_changed_paths(
              before["accumulator"], after_accumulator
          )
          state_record = {
              "verdict": (
                  "PASS"
                  if not model_changed
                  and not optimizer_changed
                  and not accumulator_changed
                  else "FAIL"
              ),
              "model_changed_paths": model_changed,
              "optimizer_changed_paths": optimizer_changed,
              "accumulator_changed_paths": accumulator_changed,
              "train_steps": self.train_steps,
          }
          state_path = os.environ.get("CANON_STATE_REPORT", "")
          if not state_path:
            raise alignment.AlignmentGateError(
                "CANON_STATE_REPORT is required for P27 gate-only"
            )
          os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
          with open(state_path, "w", encoding="utf-8") as state_file:
            json.dump(state_record, state_file, indent=2, sort_keys=True)
            state_file.write("\n")
          print(
              "[CANON_FROZENLAKE_P27] gate_state_snapshot "
              f"verdict={state_record['verdict']} model_changed="
              f"{len(model_changed)} optimizer_changed="
              f"{len(optimizer_changed)} accumulator_changed="
              f"{len(accumulator_changed)}",
              flush=True,
          )
          if state_record["verdict"] != "PASS":
            raise alignment.AlignmentGateError(
                f"P27 gate-only mutated state; report={state_path}"
            )
      if record["execution_mode"] == "update-canary":
        is_update_step = bool(
            np.asarray(jax.device_get(aux["canon/is_update_step"])).item()
        )
        before = self._canon_update_before
        if before is None:
          raise alignment.AlignmentGateError("update snapshot missing after train step")
        if not is_update_step:
          print(
              "[CANON_FROZENLAKE_P27] update_accumulation_pending "
              f"train_steps={self.train_steps}",
              flush=True,
          )
          return
        self._canon_update_before = None
        after = {
            "model": self._canon_fingerprint_state(nnx.state(self.model, nnx.Param)),
            "optimizer": self._canon_fingerprint_state(
                nnx.state(self.optimizer, nnx.optimizer.OptState)
            ),
        }
        if "accumulator" in before:
          after["accumulator"] = self._canon_fingerprint_state(
              nnx.state(self.grad_accumulator)
          )
        model_changed = self._canon_changed_paths(before["model"], after["model"])
        optimizer_changed = self._canon_changed_paths(
            before["optimizer"], after["optimizer"]
        )
        accumulator_changed = (
            self._canon_changed_paths(
                before["accumulator"], after["accumulator"]
            )
            if "accumulator" in before
            else []
        )
        update_record = {
            "verdict": (
                "PASS"
                if model_changed
                and optimizer_changed
                and not accumulator_changed
                else "FAIL"
            ),
            "alignment_hashes": record["hashes"],
            "train_steps_before": before["train_steps"],
            "expected_train_steps_after": before["train_steps"] + 1,
            "model": {
                "sampled_leaves": before["model"]["sampled_leaves"],
                "sampled_bytes": before["model"]["sampled_bytes"],
                "changed_count": len(model_changed),
                "changed_paths": model_changed,
                "before": before["model"]["leaves"],
                "after": after["model"]["leaves"],
            },
            "optimizer": {
                "sampled_leaves": before["optimizer"]["sampled_leaves"],
                "sampled_bytes": before["optimizer"]["sampled_bytes"],
                "changed_count": len(optimizer_changed),
                "changed_paths": optimizer_changed,
                "before": before["optimizer"]["leaves"],
                "after": after["optimizer"]["leaves"],
            },
            "accumulator_changed_paths": accumulator_changed,
            "checkpoint_enabled": self.config.checkpoint_root_directory is not None,
        }
        update_path = os.environ.get("CANON_UPDATE_REPORT", "")
        if not update_path:
          raise alignment.AlignmentGateError("CANON_UPDATE_REPORT is required")
        os.makedirs(os.path.dirname(update_path) or ".", exist_ok=True)
        with open(update_path, "w", encoding="utf-8") as update_file:
          json.dump(update_record, update_file, indent=2, sort_keys=True)
          update_file.write("\n")
        update_marker = (
            "[CANON_FROZENLAKE_P27] post_update_snapshot "
            if os.environ.get("CANON_FROZENLAKE_P27", "") == "1"
            else "[CANON_GSM8K_UPDATE] post_update_snapshot "
        )
        print(
            update_marker + f"verdict={update_record['verdict']} "
            f"model_changed={len(model_changed)}/"
            f"{before['model']['sampled_leaves']} "
            f"optimizer_changed={len(optimizer_changed)}/"
            f"{before['optimizer']['sampled_leaves']} "
            f"checkpoint_enabled={int(update_record['checkpoint_enabled'])}",
            flush=True,
        )
        if update_record["verdict"] != "PASS":
          raise alignment.AlignmentGateError(
              f"optimizer update did not change sampled state; report={update_path}"
          )
    assert self._buffered_train_metrics is not None
    for metric_name, op in self.rl_metrics_to_log.items():
      if metric_name not in self._buffered_train_metrics.additional_metrics:
        self._buffered_train_metrics.additional_metrics[metric_name] = (
            [aux[metric_name]],
            op,
        )
      else:
        self._buffered_train_metrics.additional_metrics[metric_name][0].append(
            aux[metric_name]
        )

  @override
  def _post_process_eval_step(self, aux: Any) -> None:
    if alignment.enabled():
      self._canon_alignment_sidecar = None
      raise alignment.AlignmentGateError(
          "evaluation is unsupported while CANON_ALIGNMENT_GATE=1; set "
          "eval_dataset=None for the one-step gate-only run"
      )
    assert self._buffered_eval_metrics is not None
    for metric_name, op in self.rl_metrics_to_log.items():
      if metric_name not in self._buffered_eval_metrics.additional_metrics:
        self._buffered_eval_metrics.additional_metrics[metric_name] = (
            [aux[metric_name]],
            op,
        )
      else:
        self._buffered_eval_metrics.additional_metrics[metric_name][0].append(
            aux[metric_name]
        )

  def _get_additional_tqdm_metrics(self) -> list[str]:
    metrics = set()
    for key_or_fn in self.tqdm_metrics_to_display:
      if isinstance(key_or_fn, str):
        metrics.add(key_or_fn)
      elif val := key_or_fn():
        metrics.add(val)
    return list(metrics)

  @property
  def _tqdm_train_metrics(self) -> list[str]:
    metrics = super()._tqdm_train_metrics
    metrics.extend(self._get_additional_tqdm_metrics())
    return metrics
