import gym
import numpy as np
from gym import spaces
import joblib
from stable_baselines3 import PPO
import pandas as pd

class Process:
    def __init__(self, pid, arrival_time, features):
        self.pid = pid
        self.arrival_time = arrival_time
        self.features = features
        self.waiting_time = 0
        self.execution_time = 0
        self.completed = False
        self.predicted_burst_time = None

class ProcessSchedulingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Load the ML burst time predictor
        self.burst_predictor = joblib.load('burst_time_predictor.joblib')
        self.scaler = joblib.load('feature_scaler.joblib')
        
        # Feature names for ML model
        self.FEATURE_NAMES = [
            'io_write_bytes',
            'num_ctx_switches_voluntary',
            'cpu_percent',
            'io_read_bytes',
            'io_read_count',
            'io_write_count'
        ]

        # Define action space (0: continue current process, 1: switch to next process)
        self.action_space = spaces.Discrete(2)

        # State space with reasonable bounds
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([100, 100, 20, 100], dtype=np.float32),
            dtype=np.float32
        )

        self.current_time = 0
        self.ready_queue = []
        self.running_process = None
        self.completed_processes = []
        self.total_processes = 0

        # Adjusted rewards for better scaling with batch size
        self.COMPLETION_REWARD = 5.0
        self.SWITCH_PENALTY = -0.5
        self.WAIT_PENALTY = -0.1
        self.TIME_QUANTUM = 4
        self.LONG_WAIT_THRESHOLD = 50  # New threshold for long waits

    def predict_burst_time(self, features):
        """Use ML model to predict burst time"""
        features_df = pd.DataFrame([features], columns=self.FEATURE_NAMES)
        scaled_features = pd.DataFrame(
            self.scaler.transform(features_df),
            columns=self.FEATURE_NAMES
        )
        return self.burst_predictor.predict(scaled_features)[0]

    def get_state(self):
        if not self.running_process:
            return np.zeros(4, dtype=np.float32)

        # Add normalized execution progress
        execution_progress = self.running_process.execution_time / max(1, self.running_process.predicted_burst_time)
        
        return np.array([
            min(100, self.running_process.predicted_burst_time),
            min(100, self.running_process.waiting_time),
            min(20, len(self.ready_queue)),
            min(100, execution_progress * 100)  # Changed to execution progress
        ], dtype=np.float32)

    def step(self, action):
        reward = 0
        done = False
        
        # Handle context switch
        if action == 1 and len(self.ready_queue) > 0:
            if self.running_process:
                # Encourage switching for long-running processes
                if self.running_process.execution_time > self.TIME_QUANTUM:
                    reward += self.SWITCH_PENALTY * 0.5  # Reduced penalty
                else:
                    reward += self.SWITCH_PENALTY
                self.ready_queue.append(self.running_process)
            self.running_process = self.ready_queue.pop(0)

        # Execute current process
        if self.running_process:
            self.running_process.execution_time += 1
            
            # Process completion
            if self.running_process.execution_time >= self.running_process.predicted_burst_time:
                self.completed_processes.append(self.running_process)
                # Scale completion reward based on waiting time
                efficiency = max(0.2, 1.0 - (self.running_process.waiting_time / self.LONG_WAIT_THRESHOLD))
                reward += self.COMPLETION_REWARD * efficiency
                
                if self.ready_queue:
                    self.running_process = self.ready_queue.pop(0)
                else:
                    self.running_process = None

        # Waiting penalty with threshold
        queue_length = len(self.ready_queue)
        for process in self.ready_queue:
            process.waiting_time += 1
            if process.waiting_time > self.LONG_WAIT_THRESHOLD:
                reward += self.WAIT_PENALTY * 2  # Double penalty for long waits
            else:
                reward += self.WAIT_PENALTY * (queue_length / 10)  # Scale with queue size

        self.current_time += 1

        # Episode completion
        if not self.running_process and not self.ready_queue:
            done = True
            if len(self.completed_processes) == self.total_processes:
                avg_waiting_time = np.mean([p.waiting_time for p in self.completed_processes])
                completion_bonus = 10.0 * (1.0 - min(1.0, avg_waiting_time / self.LONG_WAIT_THRESHOLD))
                reward += completion_bonus

        return self.get_state(), float(reward), done, {}

    def reset(self):
        """Reset environment for new episode"""
        self.current_time = 0
        self.completed_processes = []
        self.ready_queue = []
        self.running_process = None
        
        # Generate new processes
        self.total_processes = np.random.randint(5, 15)
        for i in range(self.total_processes):
            # Generate random process features
            features = [
                np.random.randint(1000, 100000),  # io_write_bytes
                np.random.randint(10, 1000),      # num_ctx_switches_voluntary
                np.random.uniform(0, 100),        # cpu_percent
                np.random.randint(1000, 100000),  # io_read_bytes
                np.random.randint(100, 1000),     # io_read_count
                np.random.randint(100, 1000)      # io_write_count
            ]
            
            process = Process(
                pid=i,
                arrival_time=self.current_time,
                features=features
            )
            
            # Predict burst time using ML model
            process.predicted_burst_time = self.predict_burst_time(features)
            self.ready_queue.append(process)

        # Start with first process
        if self.ready_queue:
            self.running_process = self.ready_queue.pop(0)

        return self.get_state()

def train_rl_scheduler(total_timesteps=150000):
    """Train the RL scheduler"""
    env = ProcessSchedulingEnv()
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0001,
        n_steps=1024,
        batch_size=32,
        n_epochs=8,
        gamma=0.99,
        verbose=1
    )
    
    try:
        model.learn(total_timesteps=total_timesteps)
        model.save("rl_scheduler_model")
        return model
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None

def evaluate_scheduler(model, episodes=10):
    """Evaluate the trained scheduler"""
    env = ProcessSchedulingEnv()
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        
        avg_waiting_time = np.mean([p.waiting_time for p in env.completed_processes])
        completion_time = env.current_time
        
        print(f"\nEpisode {episode + 1} Results:")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Average Waiting Time: {avg_waiting_time:.2f}")
        print(f"Total Completion Time: {completion_time}")
        print(f"Processes Completed: {len(env.completed_processes)}/{env.total_processes}")

if __name__ == "__main__":
    print("Training RL Scheduler...")
    model = train_rl_scheduler()
    
    print("\nEvaluating RL Scheduler...")
    evaluate_scheduler(model)