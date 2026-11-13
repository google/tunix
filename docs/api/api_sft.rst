Supervised fine-tuning (SFT)
============================

.. currentmodule:: tunix

.. autosummary::

    PeftTrainer
    TrainingConfig

    DPOTrainer
    DPOTrainingConfig
    ORPOTrainer
    ORPOTrainingConfig

    CheckpointManager
    MetricsLogger
    MetricsLoggerOptions


-------


.. autoclass:: PeftTrainer

-------

.. autoclass:: TrainingConfig

-------

.. autoclass:: DPOTrainer

-------

.. autoclass:: DPOTrainingConfig

-------

.. autoclass:: ORPOTrainer

-------

.. autoclass:: ORPOTrainingConfig

.. note::

   ``ORPOTrainingConfig`` is an alias for :class:`DPOTrainingConfig`, where
   ``algorithm`` defaults to ``"dpo"``. Pass ``algorithm="orpo"`` explicitly
   when configuring ORPO training (for example,
   ``ORPOTrainingConfig(algorithm="orpo", ...)``).

-------

.. autoclass:: CheckpointManager

-------

.. autoclass:: MetricsLogger

-------

.. autoclass:: MetricsLoggerOptions

