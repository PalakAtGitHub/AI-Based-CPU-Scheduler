import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from process_env import ProcessEnvironment

class ProcessSchedulerNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ProcessSchedulerNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size*2)
        self.fc3 = nn.Linear(hidden_size*2, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size//2)
        self.fc5 = nn.Linear(hidden_size//2, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        return self.fc5(x)

class ProcessScheduler:
    def __init__(self, state_size, action_size, learning_rate=0.0005):  # Reduced learning rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = ProcessSchedulerNetwork(state_size, action_size).to(self.device)
        self.target_net = ProcessSchedulerNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=20000)  # Increased memory size
        
        self.batch_size = 64  # Increased batch size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # Increased minimum exploration
        self.epsilon_decay = 0.997  # Slower epsilon decay
        self.target_update = 5  # More frequent target updates
        
    def get_state_tensor(self, state):
        if not state['processes']:
            return torch.zeros((1, 6)).to(self.device)
        
        process_features = torch.tensor(
            [p.features for p in state['processes']],
            dtype=torch.float32
        ).to(self.device)
        
        return process_features
    
    def select_action(self, state):
        if not state['processes']:
            return None
        
        state_tensor = self.get_state_tensor(state)
        
        if random.random() < self.epsilon:
            return random.randint(0, len(state['processes']) - 1)
        
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            return torch.argmax(q_values).item()
    
    def store_transition(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        state_tensors = [self.get_state_tensor(s) for s in states]
        next_state_tensors = [self.get_state_tensor(s) for s in next_states]
        
        state_batch = torch.cat(state_tensors)
        action_batch = torch.tensor(actions, dtype=torch.long).to(self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        next_q_values = torch.zeros(self.batch_size, dtype=torch.float32).to(self.device)
        for i, next_state in enumerate(next_states):
            if next_state['processes']:
                next_state_tensor = next_state_tensors[i]
                next_q_values[i] = self.target_net(next_state_tensor).max(1)[0].max()
        
        expected_q_values = reward_batch + self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
        
def main():
    env = ProcessEnvironment(max_processes=5)
    state_size = 6  # Size of process features
    action_size = 5  # Max number of processes
    scheduler = ProcessScheduler(state_size, action_size)
    
    num_episodes = 1000
    max_steps = 100  # Limit steps per episode
    print("Starting training...")
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}")
        print(f"Initial number of processes: {len(state['processes'])}")
        
        while steps < max_steps:
            process_idx = scheduler.select_action(state)
            if process_idx is None:
                print("No processes to schedule")
                break
            
            if process_idx >= len(state['processes']):
                print(f"Invalid process index: {process_idx}")
                break
                
            process_id = state['processes'][process_idx].id
            next_state, reward, done = env.step(process_id)
            
            print(f"Step {steps + 1}:")
            print(f"  Selected process ID: {process_id}")
            print(f"  Reward: {reward:.2f}")
            print(f"  Remaining processes: {len(next_state['processes'])}")
            
            scheduler.store_transition(state, process_idx, reward, next_state)
            loss = scheduler.train()
            
            if loss is not None:
                print(f"  Training loss: {loss:.4f}")
                print(f"  Epsilon: {scheduler.epsilon:.4f}")
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done or len(state['processes']) == 0:
                print("Episode completed!")
                break
        
        print(f"Episode {episode + 1} Summary:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Steps: {steps}")
        print(f"  Final Epsilon: {scheduler.epsilon:.4f}")
        
    torch.save(scheduler.policy_net.state_dict(), 'd:\\PROJECTS\\OS PBL\\approach2\\scheduler_model.pth')

if __name__ == "__main__":
    main()