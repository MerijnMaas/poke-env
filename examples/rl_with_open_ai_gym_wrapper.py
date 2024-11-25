import numpy as np
import tensorflow as tf
#from rl.agents.dqn_model import DQNAgent
from stable_baselines3 import DQN
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from poke_env.teambuilder import Teambuilder
from gymnasium.spaces import Box, Space
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import ObsType
from stable_baselines3.common.env_util import DummyVecEnv




from poke_env.player import Gen8EnvSinglePlayer
#from poke_env.player.random_player import player
from poke_env import RandomPlayer
 


class RandomTeamFromPool(Teambuilder):
    def __init__(self, teams):
        self.packed_teams = []

        for team in teams:
            parsed_team = self.parse_showdown_team(team)
            packed_team = self.join_team(parsed_team)
            self.packed_teams.append(packed_team)

    def yield_team(self):
        return np.random.choice(self.packed_teams)

# Definition of agent's team (Pokémon Showdown template)
OUR_TEAM = """
Pikachu-Original (M) @ Light Ball  
Ability: Static  
EVs: 252 Atk / 4 SpD / 252 Spe  
Jolly Nature  
- Volt Tackle  
- Nuzzle  
- Iron Tail  
- Knock Off  

Charizard @ Life Orb  
Ability: Solar Power  
EVs: 252 SpA / 4 SpD / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Flamethrower  
- Dragon Pulse  
- Roost  
- Sunny Day  

Blastoise @ White Herb  
Ability: Torrent  
EVs: 4 Atk / 252 SpA / 252 Spe  
Mild Nature  
- Scald  
- Ice Beam  
- Earthquake  
- Shell Smash  

Venusaur @ Black Sludge  
Ability: Chlorophyll  
EVs: 252 SpA / 4 SpD / 252 Spe  
Modest Nature  
IVs: 0 Atk  
- Giga Drain  
- Sludge Bomb  
- Sleep Powder  
- Leech Seed  

Sirfetch’d @ Aguav Berry  
Ability: Steadfast  
EVs: 248 HP / 252 Atk / 8 SpD  
Adamant Nature  
- Close Combat  
- Swords Dance  
- Poison Jab  
- Knock Off  

Tauros (M) @ Assault Vest  
Ability: Intimidate  
EVs: 252 Atk / 4 SpD / 252 Spe  
Jolly Nature  
- Double-Edge  
- Earthquake  
- Megahorn  
- Iron Head  
"""


# Definition of opponent's team (Pokémon Showdown template)

OP_TEAM = """
Eevee @ Eviolite  
Ability: Adaptability  
EVs: 252 HP / 252 Atk / 4 SpD  
Adamant Nature  
- Quick Attack  
- Flail  
- Facade  
- Wish  

Vaporeon @ Leftovers  
Ability: Hydration  
EVs: 252 HP / 252 Def / 4 SpA  
Bold Nature  
IVs: 0 Atk  
- Scald  
- Shadow Ball  
- Toxic  
- Wish  

Sylveon @ Aguav Berry  
Ability: Pixilate  
EVs: 252 HP / 252 SpA / 4 SpD  
Modest Nature  
IVs: 0 Atk  
- Hyper Voice  
- Mystical Fire  
- Psyshock  
- Calm Mind  

Jolteon @ Assault Vest  
Ability: Quick Feet  
EVs: 252 SpA / 4 SpD / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Thunderbolt  
- Hyper Voice  
- Volt Switch  
- Shadow Ball  

Leafeon @ Life Orb  
Ability: Chlorophyll  
EVs: 252 Atk / 4 SpD / 252 Spe  
Adamant Nature  
- Leaf Blade  
- Knock Off  
- X-Scissor  
- Swords Dance  

Umbreon @ Iapapa Berry  
Ability: Inner Focus  
EVs: 252 HP / 4 Atk / 252 SpD  
Careful Nature  
- Foul Play  
- Body Slam  
- Toxic  
- Wish  
"""

teams = [OUR_TEAM, OP_TEAM]

custom_builder = RandomTeamFromPool(teams)


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


# This is the function that will be used to train the dqn_model
def dqn_training(player, dqn_model, nb_steps):
    dqn_model.fit(player, nb_steps=nb_steps)
    player.complete_current_battle()


def dqn_evaluation(player, dqn_model, nb_episodes):
    # Reset battle statistics
    player.reset_battles()
    dqn_model.test(player, nb_episodes=nb_episodes, visualize=False, verbose=False)

    print(
        "DQN Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, nb_episodes)
    )




if __name__ == "__main__":
    opponent = RandomPlayer(battle_format="gen8randombattle")

    env_player = SimpleRLPlayer(battle_format="gen8randombattle", opponent=opponent)

    #opponent = RandomPlayer(battle_format="gen8randombattle")
    second_opponent = MaxDamagePlayer(battle_format="gen8randombattle")
    vec_env_player = DummyVecEnv([lambda: env_player])

    # Output dimension
    state_size = env_player.observation_space.shape[0]
    n_action = env_player.action_space.n

   

    # Our embedding have shape (1, 10), which affects our hidden layer
    # dimension and output dimension
    # Flattening resolve potential issues that would arise otherwise
    #model = Sequential([
    #Input(shape=(state_size,)),  # Input layer
    #Dense(64, activation='relu'),  # Hidden layers
    #Dense(64, activation='relu'),
    #Dense(n_action, activation='linear')  # Output layer
    #])
    #model.compile(optimizer='adam', loss='mse')


    #memory = SequentialMemory(limit=10000, window_length=1)
    
    # Ssimple epsilon greedy
    #policy = LinearAnnealedPolicy(
    #    EpsGreedyQPolicy(),
    #    attr="eps",
    #    value_max=1.0,
    #    value_min=0.05,
    #    value_test=0,
    #    nb_steps=10000,
    #)
    #print(state_size)
    #print(n_action)

    # Defining our DQN
    #dqn_model = DQN(
    #    model=model,
    #    nb_actions=env_player.action_space.n,
    #    policy=policy,
    #    memory=memory,
    #    nb_steps_warmup=1000,
    #    gamma=0.5,
    #    target_model_update=1,
    #    delta_clip=0.01,
    #    enable_double_dqn=True,
    #)

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
    dqn_model.learn(total_timesteps=1000)


    # Training
    #env_player.play_against(
    #    env_algorithm=dqn_training,
    #    opponent=opponent,
    #    env_algorithm_kwargs={"dqn_model": dqn_model, "nb_steps": NB_TRAINING_STEPS},
    #)
    #model.save("model_%d" % NB_TRAINING_STEPS)

    # Evaluation
    #print("Results against random player:")
    #env_player.play_against(
    #    env_algorithm=dqn_evaluation,
    #    opponent=opponent,
    #    env_algorithm_kwargs={"dqn_model": dqn_model, "nb_episodes": NB_EVALUATION_EPISODES},
    #)

    #print("\nResults against max player:")
    #env_player.play_against(
    #    env_algorithm=dqn_evaluation,
    #    opponent=second_opponent,
    #    env_algorithm_kwargs={"dqn_model": dqn_model, "nb_episodes": NB_EVALUATION_EPISODES},
    #)
    def evaluate_model(model, vec_env, n_episodes=100):
        victories = 0
        for episode in range(n_episodes):
            obs = vec_env.reset()  # Reset the vectorized environment
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step(action)
                done = done[0]  # Extract the "done" value from VecEnv
            # Check if the agent won
            if not info[0].get("opponent_won_battle", False):
                victories += 1
        return victories

    # Evaluate against random player
    print("Results against random player:")
    random_victories = evaluate_model(dqn_model, vec_env_player, n_episodes=100)
    print(f"DQN Evaluation: {random_victories} victories out of 100 episodes")

    # Reset the environment with a new opponent
    second_opponent = MaxDamagePlayer(battle_format="gen8randombattle")
    env_player.reset_env(restart=True, opponent=second_opponent)
    vec_env_player = DummyVecEnv([lambda: env_player])  # Wrap the new environment

    # Evaluate against max base power player
    print("Results against max base power player:")
    max_damage_victories = evaluate_model(dqn_model, vec_env_player, n_episodes=100)
    print(f"DQN Evaluation: {max_damage_victories} victories out of 100 episodes")
