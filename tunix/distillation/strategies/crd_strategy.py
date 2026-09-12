# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contrastive Representation Distillation (CRD) strategy.

Implements an InfoNCE-style contrastive loss between student and teacher
representations captured from intermediate layers via `sowed_module`.
"""

from __future__ import annotations
from typing import Any, Callable

from flax import nnx
import jax
import jax.numpy as jnp
import optax
from typing_extensions import override

from tunix.distillation.feature_extraction import sowed_module
from tunix.distillation.strategies import base_strategy
ModelForwardCallable = base_strategy.ModelForwardCallable

def _l2_normalize(
    x: jax.Array,
    axis: int = -1,
    eps: float = 1e-6,
) -> jax.Array:
    """L2-normalize along `axis` with epsilon for numerical stability."""
    norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)


def _pool_feature_to_representation(
    feat: jax.Array,
    *,
    input_mask: jax.Array | None = None,
    mask_axis: int = 1,
    eps: float = 1e-6,
) -> jax.Array:
    """Convert a single sowed feature into (B, C).

    Supported:
      (B, C) -> identity
      (B, T, C) with mask (B, T) -> masked mean
      (B, ..., C) -> mean over all non-batch, non-channel dims
    """
    if feat.ndim < 2:
        raise ValueError(
            f"Feature must have at least 2 dims (B, ...). Got {feat.shape}"
        )

    if feat.ndim == 2:
        return feat

    if input_mask is not None and feat.ndim == 3:
        if input_mask.ndim == 2 and input_mask.shape[:2] == feat.shape[:2]:
            mask = input_mask.astype(feat.dtype)[..., None]  # (B, T, 1)
            summed = jnp.sum(feat * mask, axis=mask_axis)  # (B, C)
            denom = jnp.sum(mask, axis=mask_axis)  # (B, 1)
            return summed / (denom + eps)

    reduce_axes = tuple(range(1, feat.ndim - 1))
    return jnp.mean(feat, axis=reduce_axes)


def _sowed_state_to_pooled_stack(
    sowed_state: Any,
    *,
    input_mask: jax.Array | None,
    eps: float,
) -> jax.Array:
    leaves = list(jax.tree.leaves(sowed_state))
    if not leaves:
        raise ValueError("No sowed intermediates found.")

    pooled = [
        _pool_feature_to_representation(x, input_mask=input_mask, eps=eps)
        for x in leaves
    ]  # list of (B, C)

    # Validate consistent channel dimension to allow stacking.
    c0 = int(pooled[0].shape[-1])
    for i, p in enumerate(pooled):
        if p.ndim != 2:
            raise ValueError(
                f"Expected pooled rep (B, C) but got {p.shape} at leaf {i}."
            )
        if int(p.shape[-1]) != c0:
            raise ValueError(
                "CRD currently requires all captured leaves to have the same channel "
                "dimension after pooling so they can be stacked. "
                f"Got C0={c0} and leaf[{i}].C={int(p.shape[-1])}. "
                "Suggestion: capture a single layer type, or ensure consistent dims."
            )

    return jnp.stack(pooled, axis=0)  # (N, B, C)

def _stacked_pooled_to_representation(stacked: jax.Array) -> jax.Array:
    """Convert pooled stack (N, B, C) into (B, C) by averaging N."""
    if stacked.ndim != 3:
        raise ValueError(
            f"Expected stacked pooled features (N, B, C). Got {stacked.shape}"
        )
    if stacked.shape[0] == 1:
        return stacked[0]
    return jnp.mean(stacked, axis=0)

class _ProjectionHead(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        rngs: nnx.Rngs,
        hidden_dim: int | None = None,
    ):
        self.hidden_dim = hidden_dim
        if hidden_dim is None:
            self.fc = nnx.Linear(in_dim, out_dim, rngs=rngs)
        else:
            self.fc1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
            self.fc2 = nnx.Linear(hidden_dim, out_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.hidden_dim is None:
            return self.fc(x)
        x = jax.nn.gelu(self.fc1(x))
        return self.fc2(x)


class StudentModelWithCRDHeads(nnx.Module):
    """Wrap a student model and add CRD projection heads.
    """

    def __init__(
        self,
        model: nnx.Module,
        *,
        student_rep_dim: int,
        teacher_rep_dim: int,
        embedding_dim: int,
        rngs: nnx.Rngs,
        mlp_hidden_dim: int | None = 512,
        eps: float = 1e-6,
        mask_key: str = "input_mask",
    ):
        self.model = model
        self.student_head = _ProjectionHead(
            student_rep_dim,
            embedding_dim,
            rngs=rngs,
            hidden_dim=mlp_hidden_dim,
        )
        self.teacher_head = _ProjectionHead(
            teacher_rep_dim,
            embedding_dim,
            rngs=rngs,
            hidden_dim=mlp_hidden_dim,
        )
        self.eps = eps
        self.mask_key = mask_key

    def __call__(self, *args, **kwargs) -> tuple[jax.Array, jax.Array]:
        logits = self.model(*args, **kwargs)

        s_state = sowed_module.pop_sowed_intermediate_outputs(self.model)
        if not s_state:
            raise ValueError(
                "No sowed intermediates found for student model. "
                "Did you wrap the intended layers with sowed_module?"
            )

        input_mask = kwargs.get(self.mask_key)
        s_stacked = _sowed_state_to_pooled_stack(
            s_state, input_mask=input_mask, eps=self.eps
        )
        s_rep = _stacked_pooled_to_representation(s_stacked)  # (B, C)

        z_s = _l2_normalize(self.student_head(s_rep), eps=self.eps)
        return logits, z_s

    def embed_teacher_features(
        self,
        teacher_stacked_features: jax.Array,
        *,
        input_mask: jax.Array | None = None,
    ) -> jax.Array:
        del input_mask
        t_rep = _stacked_pooled_to_representation(teacher_stacked_features)  # (B, C)
        z_t = _l2_normalize(self.teacher_head(t_rep), eps=self.eps)
        return z_t

def _infonce_loss(
    z_student: jax.Array,
    z_teacher: jax.Array,
    *,
    temperature: float,
) -> jax.Array:
    """InfoNCE with in-batch negatives: positive pairs are i->i."""
    logits = (z_student @ z_teacher.T) / temperature  # (B, B)
    bsz = logits.shape[0]
    labels = jnp.arange(bsz)
    onehot = jax.nn.one_hot(labels, bsz)
    loss = optax.softmax_cross_entropy(logits=logits, labels=onehot)
    return jnp.mean(loss)

def _setup_models_for_crd(
    student_model: nnx.Module,
    teacher_model: nnx.Module,
    *,
    student_layer_to_capture: type[nnx.Module],
    teacher_layer_to_capture: type[nnx.Module],
    dummy_student_input: dict[str, Any],
    dummy_teacher_input: dict[str, Any],
    rngs: nnx.Rngs,
    embedding_dim: int,
    mlp_hidden_dim: int | None,
    mask_key: str,
    eps: float = 1e-6,
) -> tuple[StudentModelWithCRDHeads, nnx.Module]:
    """Wrap with sowed capture, infer rep dims via dummy runs, add CRD heads."""
    sowed_module.wrap_model_with_sowed_modules(
        student_model, [student_layer_to_capture]
    )
    sowed_module.wrap_model_with_sowed_modules(
        teacher_model, [teacher_layer_to_capture]
    )

    student_model(**dummy_student_input)
    teacher_model(**dummy_teacher_input)

    s_state = sowed_module.pop_sowed_intermediate_outputs(student_model)
    t_state = sowed_module.pop_sowed_intermediate_outputs(teacher_model)

    if not s_state:
        raise ValueError(
            "No sowed intermediates found for student dummy run. "
            "Check student_layer_to_capture."
        )
    if not t_state:
        raise ValueError(
            "No sowed intermediates found for teacher dummy run. "
            "Check teacher_layer_to_capture."
        )

    s_mask = dummy_student_input.get(mask_key)
    t_mask = dummy_teacher_input.get(mask_key)

    s_stacked = _sowed_state_to_pooled_stack(
        s_state, input_mask=s_mask, eps=eps
    )  # (N, B, C)
    t_stacked = _sowed_state_to_pooled_stack(
        t_state, input_mask=t_mask, eps=eps
    )  # (N, B, C)

    s_rep = _stacked_pooled_to_representation(s_stacked)  # (B, C)
    t_rep = _stacked_pooled_to_representation(t_stacked)  # (B, C)

    if s_rep.ndim != 2 or t_rep.ndim != 2:
        raise ValueError(
            "CRD expects pooled representations to be 2D (B, C). "
            f"Got student_rep={s_rep.shape}, teacher_rep={t_rep.shape}"
        )

    wrapped_student = StudentModelWithCRDHeads(
        student_model,
        student_rep_dim=int(s_rep.shape[-1]),
        teacher_rep_dim=int(t_rep.shape[-1]),
        embedding_dim=int(embedding_dim),
        rngs=rngs,
        mlp_hidden_dim=mlp_hidden_dim,
        eps=eps,
        mask_key=mask_key,
    )
    return wrapped_student, teacher_model


def _remove_crd_from_models(
    student_model: nnx.Module,
    teacher_model: nnx.Module,
) -> tuple[nnx.Module, nnx.Module]:
    """Unwrap sowed modules and return original models."""
    if isinstance(student_model, StudentModelWithCRDHeads):
        base_student = student_model.model
    else:
        base_student = student_model

    sowed_module.unwrap_sowed_modules(base_student)
    sowed_module.unwrap_sowed_modules(teacher_model)
    return base_student, teacher_model

class ContrastiveRepresentationDistillationStrategy(base_strategy.BaseStrategy):
    """CRD: contrastive loss between student/teacher intermediates."""
    def __init__(
        self,
        student_forward_fn: ModelForwardCallable[Any],
        teacher_forward_fn: ModelForwardCallable[Any],
        labels_fn: Callable[..., jax.Array],
        *,
        student_layer_to_capture: type[nnx.Module],
        teacher_layer_to_capture: type[nnx.Module],
        dummy_student_input: dict[str, jax.Array],
        dummy_teacher_input: dict[str, jax.Array],
        rngs: nnx.Rngs,
        embedding_dim: int = 128,
        mlp_hidden_dim: int | None = 512,
        temperature: float = 0.2,
        alpha: float = 0.75,
        symmetric: bool = False,
        mask_key: str = "input_mask",
        eps: float = 1e-6,
    ):
        super().__init__(student_forward_fn, teacher_forward_fn, labels_fn)

        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.student_layer_to_capture = student_layer_to_capture
        self.teacher_layer_to_capture = teacher_layer_to_capture
        self.dummy_student_input = dummy_student_input
        self.dummy_teacher_input = dummy_teacher_input
        self.rngs = rngs
        self.embedding_dim = int(embedding_dim)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.symmetric = bool(symmetric)
        self.mask_key = mask_key
        self.eps = float(eps)

    @override
    def pre_process_models(
        self,
        student_model: nnx.Module,
        teacher_model: nnx.Module,
    ) -> tuple[nnx.Module, nnx.Module]:
        return _setup_models_for_crd(
            student_model,
            teacher_model,
            student_layer_to_capture=self.student_layer_to_capture,
            teacher_layer_to_capture=self.teacher_layer_to_capture,
            dummy_student_input=self.dummy_student_input,
            dummy_teacher_input=self.dummy_teacher_input,
            rngs=self.rngs,
            embedding_dim=self.embedding_dim,
            mlp_hidden_dim=self.mlp_hidden_dim,
            mask_key=self.mask_key,
            eps=self.eps,
        )

    @override
    def post_process_models(
        self,
        student_model: nnx.Module,
        teacher_model: nnx.Module,
    ) -> tuple[nnx.Module, nnx.Module]:
        return _remove_crd_from_models(student_model, teacher_model)

    @override
    def get_teacher_outputs(
        self,
        teacher_model: nnx.Module,
        inputs: dict[str, jax.Array],
    ) -> jax.Array:
        self._teacher_forward_fn(teacher_model, **inputs)

        t_state = sowed_module.pop_sowed_intermediate_outputs(teacher_model)
        if not t_state:
            raise ValueError(
                "No sowed intermediates found for teacher forward pass. "
                "Did you wrap the intended layers with sowed_module?"
            )

        # Produce pooled stacked features (N, B, C) and stop gradient.
        input_mask = inputs.get(self.mask_key)
        t_stacked = _sowed_state_to_pooled_stack(
            t_state, input_mask=input_mask, eps=self.eps
        )
        return jax.lax.stop_gradient(t_stacked)

    @override
    def get_train_loss(
        self,
        student_model: nnx.Module,
        teacher_output: jax.Array,
        inputs: dict[str, jax.Array],
    ) -> jax.Array:
        if not isinstance(student_model, StudentModelWithCRDHeads):
            raise TypeError(
                "CRD expects student_model to be StudentModelWithCRDHeads. "
                "Did pre_process_models run?"
            )

        student_logits, z_s = self._student_forward_fn(student_model, **inputs)

        z_t = student_model.embed_teacher_features(
            jax.lax.stop_gradient(teacher_output)
        )

        crd_loss = _infonce_loss(z_s, z_t, temperature=self.temperature)
        if self.symmetric:
            crd_loss = 0.5 * (
                crd_loss + _infonce_loss(z_t, z_s, temperature=self.temperature)
            )

        labels = self._labels_fn(**inputs)
        task_loss = jnp.mean(
            optax.softmax_cross_entropy(logits=student_logits, labels=labels)
        )
        return (self.alpha * crd_loss) + ((1.0 - self.alpha) * task_loss)

    @override
    def get_eval_loss(
        self,
        student_model: nnx.Module,
        inputs: dict[str, jax.Array],
    ) -> jax.Array:
        out = self._student_forward_fn(student_model, **inputs)
        student_logits = out[0] if isinstance(out, (tuple, list)) else out
        labels = self._labels_fn(**inputs)
        return jnp.mean(
            optax.softmax_cross_entropy(logits=student_logits, labels=labels)
        )

    def compute_loss(
        self,
        student_output: Any,
        teacher_output: Any,
        labels: jax.Array,
    ) -> jax.Array:
        raise NotImplementedError("CRD uses get_train_loss override.")

    def compute_eval_loss(
        self,
        student_output: Any,
        labels: jax.Array,
    ) -> jax.Array:
        raise NotImplementedError("CRD uses get_eval_loss override.")