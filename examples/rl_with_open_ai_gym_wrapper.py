import numpy as np
import tensorflow as tf
from stable_baselines3 import DQN
from gymnasium.spaces import Box, Space
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import ObsType
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy




from poke_env.player import Gen8EnvSinglePlayer, SimpleHeuristicsPlayer
#from poke_env.player.random_player import player
from poke_env import RandomPlayer
 





# We define our RL player
# It needs a state embedder and a reward computer, hence these two methods
class SimpleRLPlayer(Gen8EnvSinglePlayer):
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

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [remaining_mon_team, remaining_mon_opponent],
            ]
        )
    
    def reward_computing_helper(
            self,
            battle: AbstractBattle,
            *,
            fainted_value: float = 0.15,
            hp_value: float = 0.15,
            number_of_pokemons: int = 6,
            starting_value: float = 0.0,
            status_value: float = 0.15,
            victory_value: float = 1.0
    ) -> float:
        # 1st compute
        if battle not in self._reward_buffer:
            self._reward_buffer[battle] = starting_value
        current_value = 0

        # Verify if pokemon have fainted or have status
        for mon in battle.team.values():
            current_value += mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value -= fainted_value
            elif mon.status is not None:
                current_value -= status_value

        current_value += (number_of_pokemons - len(battle.team)) * hp_value

        # Verify if opponent pokemon have fainted or have status
        for mon in battle.opponent_team.values():
            current_value -= mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value += fainted_value
            elif mon.status is not None:
                current_value += status_value

        current_value -= (number_of_pokemons - len(battle.opponent_team)) * hp_value

        # Verify if we won or lost
        if battle.won:
            current_value += victory_value
        elif battle.lost:
            current_value -= victory_value

        # Value to return
        to_return = current_value - self._reward_buffer[battle]
        self._reward_buffer[battle] = current_value
        return to_return

    def calc_reward(self, last_battle, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=2, hp_value=1, victory_value=30
        )
    def describe_embedding(self) -> Space:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )



class MaxDamagePlayer(RandomPlayer):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)


NB_TRAINING_STEPS = 10000
NB_EVALUATION_EPISODES = 100

tf.random.set_seed(0)
np.random.seed(0)







if __name__ == "__main__":
    opponent = RandomPlayer(battle_format="gen8randombattle")
    second_opponent = MaxDamagePlayer(battle_format="gen8randombattle")
    third_opponent = SimpleHeuristicsPlayer(battle_format="gen8randombattle")
    
    
    env_player = SimpleRLPlayer(battle_format="gen8randombattle", opponent=opponent, start_challenging=True)
    #eval_env = SimpleRLPlayer(battle_format="gen8randombattle", opponent=opponent, start_challenging=True)
    #second_eval_env = SimpleRLPlayer(battle_format="gen8randombattle", opponent=second_opponent, start_challenging=True)
    #third_eval_env = SimpleRLPlayer(battle_format="gen8randombattle", opponent=third_opponent, start_challenging=True)
    
    

    
    vec_env_player = DummyVecEnv([lambda: env_player])
    #vec_eval_env = DummyVecEnv([lambda: eval_env])
    #vec_second_eval_env = DummyVecEnv([lambda: second_eval_env])
    #vec_third_eval_env = DummyVecEnv([lambda: third_eval_env])
    # Output dimension
    state_size = env_player.observation_space.shape[0]
    n_action = env_player.action_space.n

   

    

        # Initialize the DQN model
    dqn_model = DQN(
        policy="MlpPolicy",           # Automatically defines a feedforward network
        env=vec_env_player,               # Your environment
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

    #dqn_model.compile(Adam(lr=0.00025), metrics=["mae"])
    dqn_model.learn(total_timesteps=5000)
    dqn_model.save('DQNmodel')
    del dqn_model
    vec_env_player.close()

    dqn_model = DQN.load("DQNmodel", env=vec_env_player)

    
    
    
    eval_env = SimpleRLPlayer(battle_format="gen8randombattle", opponent=opponent, start_challenging=True)
    vec_eval_env = DummyVecEnv([lambda: eval_env])
    evaluate_policy(dqn_model, vec_eval_env, n_eval_episodes=50)
    print(f"against Random OPP: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes")
    vec_eval_env.close()

    #second_opponent = MaxDamagePlayer(battle_format="gen8randombattle")
    second_eval_env = SimpleRLPlayer(battle_format="gen8randombattle", opponent=second_opponent, start_challenging=True)
    vec_second_eval_env = DummyVecEnv([lambda: second_eval_env])
    evaluate_policy(dqn_model, vec_second_eval_env, n_eval_episodes=50)
    print(f"against MaxDamage OPP: {second_eval_env.n_won_battles} victories out of {second_eval_env.n_finished_battles} episodes")
    vec_second_eval_env.close()

    #third_opponent = SimpleHeuristicsPlayer(battle_format="gen8randombattle")
    third_eval_env = SimpleRLPlayer(battle_format="gen8randombattle", opponent=third_opponent, start_challenging=True)
    vec_third_eval_env = DummyVecEnv([lambda: third_eval_env])
    evaluate_policy(dqn_model, vec_third_eval_env, n_eval_episodes=50)
    print(f"against Simpleheuristics OPP: {third_eval_env.n_won_battles} victories out of {third_eval_env.n_finished_battles} episodes")
    vec_third_eval_env.close()