from random import sample
from collections import deque
import numpy as np
import torch
import torch.nn as nn

# Check for GPU availability (CUDA first, then CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


class NeuralNet(torch.nn.Module):
    """
    Implements a neural network representation of
    the Q-function for use in DQN.
    """

    def __init__(self):
        super(NeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(60, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        """
        Input represents state/observation space encoding
        Output is a q-function estimate for each possible
        discrete action.
        """
        logits = self.net(x)
        return logits


class BusAgent:
    """
    Implements a deep Q-learning agent for the C1 Bus environment.
    """

    def __init__(
        self,
        env,
        discount=0.95,
        learning_rate=0.0005,
        buffer_size=100000,
        batch_size=128,
        target_update_freq=5000,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.999,
    ):
        """
        Initialize the DQN agent.

        Args:
            discount: Discount factor (gamma)
            learning_rate: Learning rate for optimizer
            buffer_size: Maximum size of replay buffer
            batch_size: Number of transitions to sample per update
            target_update_freq: Steps between target network updates
            epsilon_start: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Multiplicative decay for epsilon
        """
        self.env = env
        self.discount = discount
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Initialize replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)

        # Initialize networks, loss, and optimizer
        self.main_q = NeuralNet().to(device)
        self.target_q = NeuralNet().to(device)

        self.target_q.load_state_dict(self.main_q.state_dict())
        self.target_q.eval()
        self.step_count = 0

        self.optimizer = torch.optim.Adam(self.main_q.parameters(), lr=learning_rate, weight_decay=1e-5)
        # self.loss = nn.MSELoss()
        self.loss = nn.SmoothL1Loss()  # Huber loss, prefered loss function

    def action_select(self, state):
        """
        Epsilon-greedy action selection using neural network.

        Args:
            state: NumPy array of shape (8,)

        Returns:
            action: Integer action (0-3)
        """
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = self.main_q(state_tensor)

        return torch.argmax(q_values, dim=1).item()

    def update(self, state, action, reward, next_state, terminated):
        """
        Store experience and perform learning update if buffer is ready.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            terminated: Whether episode terminated
        """
        self.replay_buffer.append((state, action, reward, next_state, terminated))

        if len(self.replay_buffer) < self.batch_size:
            return

        batch = sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float).to(device)
        actions = (
            torch.tensor(np.array(actions), dtype=torch.long).unsqueeze(1).to(device)
        )
        rewards = (
            torch.tensor(np.array(rewards), dtype=torch.float).unsqueeze(1).to(device)
        )
        next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(device)
        terminations = (
            torch.tensor(np.array(dones), dtype=torch.int).unsqueeze(1).to(device)
        )

        q_values = self.main_q(states).gather(1, actions)

        with torch.no_grad():
            max_next_q = self.target_q(next_states).max(1, keepdim=True)[0]
            targets = rewards + (1 - terminations) * self.discount * max_next_q

        loss = self.loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(self.main_q.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay) # epsilon decay after each update step

        self.step_count += 1

        if self.step_count % self.target_update_freq == 0:
            self.target_q.load_state_dict(self.main_q.state_dict())

    def render(self, mode="human"):
        """
        AI Attribution: a significant portion of this rendering code was inspire by LLM output
        
        Render the current state of the bus route and buses using ASCII art.
        Rows represent stops, columns represent buses.
        
        Args:
            mode: Render mode ("human" for console output)
        """
        if mode != "human":
            raise ValueError(f"Render mode {mode} not supported. Use 'human'.")
        
        env = self.env.unwrapped  # Access the underlying environment for rendering
        num_stops = env.num_stops
        max_buses = env.max_buses
        
        # Create the route visualization
        print("\n" + "=" * 120)
        print(f"Time Step: {env.timestep:4d} | Active Buses: {env._active_bus_count():2d} | "
              f"Total Waiting: {int(np.sum(env.queues)):4d}")
        print("=" * 120)
        
        # Header row with bus numbers
        header = " Stop | Seg | Queue |"
        for bus_idx in range(max_buses):
            header += f" Bus{bus_idx} |"
        print(header)
        print("-" * len(header))
        
        # Display each stop as a row
        for stop_idx in range(num_stops):
            segment = "WB" if stop_idx < 8 else "EB"
            queue_len = int(env.queues[stop_idx])
            
            # Format queue column
            if queue_len == 0:
                queue_str = "  ·  "
            elif queue_len < 100:
                queue_str = f" {queue_len:2d}  "
            else:
                queue_str = f" {queue_len:3d} "
            
            stop_line = f"  {stop_idx:2d}  |  {segment} |{queue_str}|"
            
            # Add bus info for each bus
            for bus_idx in range(max_buses):
                bus = env.buses[bus_idx]
                if bus["active"] and bus["stop"] == stop_idx:
                    occ = int(bus["occupancy"])
                    held_char = "H" if bus["held"] else ""
                    stop_line += f" {occ:2d}{held_char}  |"
                else:
                    stop_line += "  -  |"
            
            print(stop_line)
        
        print("=" * 120)
        print("Legend: ##· = occupancy (· = moving)  |  ##H = occupancy (H = held)  |  - = bus not at this stop")
        print()
