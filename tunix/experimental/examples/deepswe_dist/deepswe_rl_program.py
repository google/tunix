import asyncio
from typing import Any
import logging
from tunix.experimental.orchestrator.rl_program import RLProgram, RLStepResult
from examples.deepswe.swe_env import SWEEnv, _init_global_fleet
from examples.deepswe.swe_agent import SWEAgent

class DeepsweRLProgram(RLProgram):
    def __init__(self, dataset, algo, max_steps: int = 100):
        super().__init__()
        self.dataset = dataset
        self.algo = algo
        self.max_steps = max_steps
        
        # We will reuse the same environment config logic as the standalone version
        self.agent_kwargs = {}
        self.env_kwargs = {"max_steps": 10, "step_timeout": 1800, "backend": "kubernetes"}

    async def _run_agent_sandbox_loop(self, example):
        # 1. Initialize the agent and environment for this specific task
        agent = SWEAgent(system_prompt="You are an expert software engineer... (stub)", **self.agent_kwargs)
        env = SWEEnv(example, group_id=0, pair_index=0, **self.env_kwargs)
        
        # 2. Extract initial variables
        prompt = agent.build_prompt(example)
        observation, reward, done = env.reset()
        
        chat_history = prompt + " " + observation
        
        # 3. Main Multi-turn execution loop
        while not done:
            # THIS is the distributed magic! 
            # We call our remote Rollout pod transparently using self.engine.generator
            action = await self.engine.generator.generate_async([chat_history])
            action = action[0] # Unwrap batch
            
            # Feed the generated bash command back into the Kubernetes sandbox!
            observation, reward, done = await env.step(action)
            chat_history = chat_history + " " + action + " " + observation
            
        return chat_history, reward

    async def run(self):
        logging.info("Starting DeepSWE RL Program loop...")
        
        # In a real run, we iterate over the dataset, run concurrent tasks, and accumulate batches
        for step in range(self.max_steps):
            example = next(self.dataset)
            
            # Trigger the multi-turn sandbox run asynchronously
            trajectory, reward = await self._run_agent_sandbox_loop(example)
            
            # Once the environment finishes, send the trajectory to the Trainer!
            await self.engine.trainer.train_step_async(
                prompts=[trajectory], # simplified 
                responses=[""], 
                rewards=[reward]
            )
            
            logging.info(f"Step {step} finished. Reward: {reward}")
            self.last_step_result = RLStepResult(
                step=step,
                policy_version=step,
                num_rollouts=1,
                num_microbatches=1,
                reward_mean=reward,
                reward_std=0.0
            )

        self._is_running = False
