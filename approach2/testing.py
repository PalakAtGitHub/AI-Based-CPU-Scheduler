import torch
from RL import ProcessScheduler, ProcessEnvironment
import matplotlib.pyplot as plt
import numpy as np

def evaluate_scheduler(scheduler, env, num_test_episodes=100):
    total_rewards = []
    waiting_times = []
    completion_times = []
    throughput = []
    
    print("\nStarting Evaluation...")
    scheduler.epsilon = 0.0  # Disable exploration for testing
    
    for episode in range(num_test_episodes):
        state = env.reset()
        episode_reward = 0
        completed = 0
        steps = 0
        
        while steps < 100:  # Max steps per episode
            process_idx = scheduler.select_action(state)
            if process_idx is None or process_idx >= len(state['processes']):
                break
                
            process_id = state['processes'][process_idx].id
            next_state, reward, done = env.step(process_id)
            
            episode_reward += reward
            completed += 1 if reward > 0 else 0
            steps += 1
            state = next_state
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        waiting_times.append(sum(p.waiting_time for p in env.completed_processes))
        completion_times.append(steps)
        throughput.append(completed / steps if steps > 0 else 0)
    
    # Calculate metrics
    metrics = {
        'avg_reward': np.mean(total_rewards),
        'avg_waiting': np.mean(waiting_times),
        'avg_completion': np.mean(completion_times),
        'avg_throughput': np.mean(throughput),
        'success_rate': sum(1 for r in total_rewards if r > 0) / len(total_rewards)
    }
    
    print("\nTest Results:")
    print(f"Average Reward: {metrics['avg_reward']:.2f}")
    print(f"Average Waiting Time: {metrics['avg_waiting']:.2f}")
    print(f"Average Completion Time: {metrics['avg_completion']:.2f}")
    print(f"Average Throughput: {metrics['avg_throughput']:.2f}")
    print(f"Success Rate: {metrics['success_rate']:.2%}")
    
    return metrics

def plot_results(metrics_list):
    episodes = range(len(metrics_list))
    rewards = [m['avg_reward'] for m in metrics_list]
    waiting = [m['avg_waiting'] for m in metrics_list]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(episodes, rewards, 'b-')
    plt.title('Average Rewards')
    plt.xlabel('Test Run')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(episodes, waiting, 'r-')
    plt.title('Average Waiting Time')
    plt.xlabel('Test Run')
    plt.ylabel('Time')
    
    plt.tight_layout()
    plt.savefig('scheduler_performance.png')
    plt.show()

def main():
    # Load the trained model
    env = ProcessEnvironment(max_processes=5)
    state_size = 6
    action_size = 5
    scheduler = ProcessScheduler(state_size, action_size)
    
    try:
        scheduler.policy_net.load_state_dict(torch.load('d:\\PROJECTS\\OS PBL\\approach2\\scheduler_model.pth'))
        print("Loaded trained model successfully")
    except:
        print("No trained model found. Please train the model first.")
        return
    
    # Run multiple test sessions
    metrics_history = []
    num_test_runs = 5
    
    for i in range(num_test_runs):
        print(f"\nTest Run {i+1}/{num_test_runs}")
        metrics = evaluate_scheduler(scheduler, env)
        metrics_history.append(metrics)
    
    # Plot results
    plot_results(metrics_history)

if __name__ == "__main__":
    main()