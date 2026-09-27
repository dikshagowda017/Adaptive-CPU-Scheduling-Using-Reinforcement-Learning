# Adaptive CPU Scheduling Using Reinforcement Learning

A Reinforcement Learning based CPU scheduling project that uses **Q-Learning** to make adaptive CPU scheduling decisions. The learned approach is evaluated alongside traditional scheduling algorithms such as **FCFS, Round Robin, and Priority Scheduling**.

## Project Overview

Traditional CPU scheduling algorithms use fixed rules to select processes for execution. This project explores the use of **Reinforcement Learning** to learn scheduling decisions based on the current state of the CPU ready queue.

The project generates synthetic workloads and provides a simulation environment where a Q-Learning agent learns scheduling policies.

## Features

- Synthetic CPU workload generation
- Poisson-based process arrival times
- Exponential CPU burst-time generation
- Random process priorities
- FCFS scheduling
- Round Robin scheduling
- Priority scheduling
- Q-Learning based CPU scheduling
- Gymnasium-compatible environment
- Epsilon-greedy action selection
- State discretization for tabular Q-Learning
- Waiting-time based reward
- Context-switch penalty
- Fairness-based reward
- Process-completion bonus
- Performance metric evaluation

## Project Structure

```text
Adaptive-CPU-Scheduling-Using-Reinforcement-Learning/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── workload_generator.py
├── simulator.py
├── fcfs.py
├── round_robin.py
├── priority.py
├── environment.py
└── q_learning.py
