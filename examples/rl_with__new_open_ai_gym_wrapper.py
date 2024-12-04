import asyncio
import tensorflow 
#tf.keras.__version__ = __version__
import keras
from keras import models
import random

import numpy as np
from gymnasium.spaces import Box, Space
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import DQN
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from tabulate import tabulate
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from poke_env.teambuilder import Teambuilder
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy











from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import (
    Gen8EnvSinglePlayer,
    MaxBasePowerPlayer,
    ObsType,
    RandomPlayer,
    SimpleHeuristicsPlayer,
    background_cross_evaluate,
    background_evaluate_player,
)






class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle) -> ObsType:
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = battle.opponent_active_pokemon.damage_multiplier(move)
                

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)

    def describe_embedding(self) -> Space:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )


async def main():
    # First test the environment to ensure the class is consistent
    # with the OpenAI API
    opponent = RandomPlayer(battle_format="gen8anythinggoes")
    test_env = SimpleRLPlayer(
        battle_format="gen8anythinggoes", start_challenging=True, opponent=opponent
    )
    check_env(test_env)
    #debug this line
    test_env.close()

    # Create one environment for training and one for evaluation
    opponent = RandomPlayer(battle_format="gen8anythinggoes")
    train_env = SimpleRLPlayer(
        battle_format="gen8anythinggoes", opponent=opponent, start_challenging=True
    )
    vec_env_train = DummyVecEnv([lambda: train_env])
    opponent = RandomPlayer(battle_format="gen8anythinggoes")
    eval_env = SimpleRLPlayer(
        battle_format="gen8anythinggoes", opponent=opponent, start_challenging=True
    )
    vec_env_eval = DummyVecEnv([lambda: eval_env])


    # Compute dimensions
    n_action = train_env.action_space.n
    input_shape = (1,) + train_env.observation_space.shape

    

    

    dqn_model2 = DQN(
        policy="MlpPolicy",           # Automatically defines a feedforward network
        env=vec_env_train,               # Your environment
        learning_rate=0.00025,        # Learning rate for Adam optimizer
        buffer_size=10000,            # Replay buffer size
        learning_starts=1000,         # Warm-up steps before training starts
        gamma=0.5,                    # Discount factor
        target_update_interval=1000,  # Steps between target network updates
        exploration_fraction=0.1,     # Fraction of training steps with exploration
        exploration_final_eps=0.05,   # Final epsilon for exploration
        exploration_initial_eps=1.0,  # Initial epsilon for exploration
        train_freq=4,                 # Training frequency
        gradient_steps=1,             # Gradient steps per training iteration
        policy_kwargs=dict(net_arch=[64, 64]),  # Specify neural network architecture
        verbose=1                     # Logging verbosity
    )
    dqn_model2.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    # Training the model
    dqn_model2.fit(train_env, nb_steps=10000)
    train_env.close()

    # Evaluating the model
    print("Results against random player:")
    dqn_model2.test(eval_env, nb_episodes=100, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )
    second_opponent = MaxBasePowerPlayer(battle_format="gen8anythinggoes")
    eval_env.reset_env(restart=True, opponent=second_opponent)
    print("Results against max base power player:")
    dqn_model2.test(eval_env, nb_episodes=100, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )
    eval_env.reset_env(restart=False)

    # Evaluate the player with included util method
    n_challenges = 250
    placement_battles = 40
    eval_task = background_evaluate_player(
        eval_env.agent, n_challenges, placement_battles
    )
    dqn_model2.test(eval_env, nb_episodes=n_challenges, verbose=False, visualize=False)
    print("Evaluation with included method:", eval_task.result())
    eval_env.reset_env(restart=False)

    # Cross evaluate the player with included util method
    n_challenges = 50
    players = [
        eval_env.agent,
        RandomPlayer(battle_format="gen8anythinggoes"),
        MaxBasePowerPlayer(battle_format="gen8anythinggoes"),
        SimpleHeuristicsPlayer(battle_format="gen8anythinggoes"),
    ]
    cross_eval_task = background_cross_evaluate(players, n_challenges)
    dqn_model2.test(
        eval_env,
        nb_episodes=n_challenges * (len(players) - 1),
        verbose=False,
        visualize=False,
    )
    cross_evaluation = cross_eval_task.result()
    table = [["-"] + [p.username for p in players]]
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])
    print("Cross evaluation of DQN with baselines:")
    print(tabulate(table))
    eval_env.close()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())

