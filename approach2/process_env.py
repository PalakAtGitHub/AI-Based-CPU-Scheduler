import numpy as np
from dataclasses import dataclass
from typing import List, Dict
import random

@dataclass
class Process:
    id: int
    burst_time: float
    arrival_time: float
    priority: int
    cpu_usage: float
    memory_usage: float
    io_intensity: float
    remaining_time: float
    waiting_time: float = 0
    
    @property
    def features(self) -> np.ndarray:
        return np.array([
            self.cpu_usage,
            self.memory_usage,
            self.io_intensity,
            self.remaining_time / self.burst_time,
            self.waiting_time,
            self.priority
        ])

class ProcessEnvironment:
    def __init__(self, max_processes=10):
        self.max_processes = max_processes
        self.current_time = 0
        self.processes: List[Process] = []
        self.completed_processes = []
        self.next_process_id = 0
        
    def generate_process(self) -> Process:
        burst_time = random.uniform(1, 10)
        return Process(
            id=self.next_process_id,
            burst_time=burst_time,
            arrival_time=self.current_time,
            priority=random.randint(1, 5),
            cpu_usage=random.uniform(0, 1),
            memory_usage=random.uniform(0, 1),
            io_intensity=random.uniform(0, 1),
            remaining_time=burst_time
        )
    
    def step(self, selected_process_id: int) -> tuple:
        if not self.processes:
            return None, 0, True
        
        # Execute selected process
        selected_process = None
        for proc in self.processes:
            if proc.id == selected_process_id:
                selected_process = proc
                break
        
        if selected_process is None:
            return self.get_state(), -1, False
        
        # Update process times
        time_slice = min(1.0, selected_process.remaining_time)
        selected_process.remaining_time -= time_slice
        self.current_time += time_slice
        
        # Update waiting times for other processes
        for proc in self.processes:
            if proc.id != selected_process_id:
                proc.waiting_time += time_slice
        
        # Remove completed process
        if selected_process.remaining_time <= 0:
            self.processes.remove(selected_process)
            self.completed_processes.append(selected_process)
        
        # Add new processes randomly
        if len(self.processes) < self.max_processes and random.random() < 0.3:
            self.processes.append(self.generate_process())
            self.next_process_id += 1
        
        # Calculate reward
        reward = self.calculate_reward(selected_process, time_slice)
        
        # Check if episode is done
        done = len(self.processes) == 0
        
        return self.get_state(), reward, done
    
    def get_state(self) -> Dict:
        if not self.processes:
            return {
                'processes': [],
                'contention': {
                    'cpu_contention': 0.0,
                    'memory_contention': 0.0,
                    'io_contention': 0.0,
                    'overall_contention': 0.0
                },
                'temporal': {'recent_load': [0.0] * 10}
            }
        
        # Calculate system contention
        cpu_usage = sum(p.cpu_usage for p in self.processes) / self.max_processes
        memory_usage = sum(p.memory_usage for p in self.processes) / self.max_processes
        io_usage = sum(p.io_intensity for p in self.processes) / self.max_processes
        
        return {
            'processes': self.processes,
            'contention': {
                'cpu_contention': cpu_usage,
                'memory_contention': memory_usage,
                'io_contention': io_usage,
                'overall_contention': (cpu_usage + memory_usage + io_usage) / 3
            },
            'temporal': {'recent_load': [cpu_usage] * 10}
        }
    
    def calculate_reward(self, process: Process, time_slice: float) -> float:
        if process is None:
            return -1.0
        
        completion_reward = 1.0 if process.remaining_time <= 0 else 0.0
        waiting_penalty = -0.1 * process.waiting_time
        priority_factor = 0.2 * process.priority
        
        return completion_reward + waiting_penalty + priority_factor
    
    def reset(self):
        self.current_time = 0
        self.processes = []
        self.completed_processes = []
        self.next_process_id = 0
        
        # Add initial processes
        num_initial = random.randint(1, self.max_processes)
        for _ in range(num_initial):
            self.processes.append(self.generate_process())
            self.next_process_id += 1
        
        return self.get_state()