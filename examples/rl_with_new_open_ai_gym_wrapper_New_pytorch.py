import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gymnasium.spaces import Box, Space
from gymnasium.utils.env_checker import check_env
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

# Define a custom player class inheriting from Gen8EnvSinglePlayer
class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        """Calculate the reward based on the battle outcome."""
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle) -> ObsType:
        """Convert the current battle state into a numerical observation."""
        moves_base_power = -np.ones(4)  # Default base power of moves
        moves_dmg_multiplier = np.ones(4)  # Default damage multiplier

        # Extract move details for the available moves
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power / 100  # Normalize base power
            if move.type:
                moves_dmg_multiplier[i] = battle.opponent_active_pokemon.damage_multiplier(move)

        # Calculate team stats (number of fainted Pokémon)
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6

        # Final observation vector
        final_vector = np.concatenate(
            [moves_base_power, moves_dmg_multiplier, [fainted_mon_team, fainted_mon_opponent]]
        )
        return np.float32(final_vector)

    def describe_embedding(self) -> Space:
        """Define the space for observations."""
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )


# Define the PyTorch DQN network
class DQNNetwork(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(DQNNetwork, self).__init__()
        # Define the layers of the network
        self.fc1 = nn.Linear(input_dim, 128)  # Input layer
        self.fc2 = nn.Linear(128, 64)  # Hidden layer
        self.fc3 = nn.Linear(64, n_actions)  # Output layer

    def forward(self, x):
        """Define the forward pass."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Define the DQN agent class
class DQNAgent:
    def __init__(self, model, n_actions, input_dim, memory_size, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, lr=0.00025):
        self.model = model  # Main network
        self.target_model = DQNNetwork(input_dim, n_actions)  # Target network
        self.n_actions = n_actions  # Number of actions
        self.memory = []  # Replay memory
        self.memory_size = memory_size
        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.epsilon_decay = epsilon_decay  # Rate at which exploration decreases
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)  # Optimizer
        self.loss_fn = nn.MSELoss()  # Loss function

    def act(self, state):
        """Choose an action based on the current state."""
        if np.random.rand() <= self.epsilon:
            # Explore: choose a random action
            return np.random.choice(self.n_actions)
        # Exploit: choose the action with the highest predicted Q-value
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        """Store the experience in replay memory."""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)  # Remove the oldest memory if the buffer is full
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """Train the network using replay memory."""
        if len(self.memory) < batch_size:
            return  # Not enough experiences to sample from
        minibatch = np.random.choice(self.memory, batch_size, replace=False)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            target = reward
            if not done:
                # Update target with discounted future reward
                target += self.gamma * torch.max(self.target_model(next_state_tensor)).item()
            target_f = self.model(state_tensor)
            target_f[action] = target
            loss = self.loss_fn(target_f, self.model(state_tensor))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_target_model(self):
        """Update the target model to match the main model."""
        self.target_model.load_state_dict(self.model.state_dict())


# Define the main training and evaluation loop
async def main():
    # Set up opponents and environments
    opponent = RandomPlayer(battle_format="gen8randombattle")
    train_env = SimpleRLPlayer(battle_format="gen8randombattle", opponent=opponent, start_challenging=True)
    eval_env = SimpleRLPlayer(battle_format="gen8randombattle", opponent=opponent, start_challenging=True)

    # Define observation space and action space
    input_dim = train_env.observation_space.shape[0]
    n_actions = train_env.action_space.n

    # Initialize the DQN network and agent
    model = DQNNetwork(input_dim, n_actions)
    agent = DQNAgent(model, n_actions, input_dim=input_dim, memory_size=10000)

    # Training loop
    for episode in range(1000):
        state = train_env.reset()
        done = False
        while not done:
            action = agent.act(state)  # Select an action
            next_state, reward, done, _ = train_env.step(action)  # Execute action
            agent.remember(state, action, reward, next_state, done)  # Store experience
            state = next_state  # Transition to next state
        agent.replay(batch_size=32)  # Train using replay memory
        agent.update_target_model()  # Update the target model periodically

    print("Training complete. Evaluating...")
    eval_env.close()


if __name__ == "__main__":
    asyncio.run(main())
