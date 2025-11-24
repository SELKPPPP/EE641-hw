"""
Training script for Value Iteration and Q-Iteration.
"""

import numpy as np
import argparse
import json
import os
from environment import GridWorldEnv
from value_iteration import ValueIteration
from q_iteration import QIteration


def main():
    """
    Run both algorithms and save results.
    """
    parser = argparse.ArgumentParser(description='Train RL algorithms on GridWorld')
    parser.add_argument('--seed', type=int, default=641, help='Random seed')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=1e-4, help='Convergence threshold')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum iterations')
    args = parser.parse_args()

    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/visualizations', exist_ok=True)

    # TODO: Initialize environment with seed
    print(f"Initializing environment with seed {args.seed}")
    env = GridWorldEnv(seed=args.seed)

    # TODO: Run Value Iteration
    #       - Create ValueIteration solver
    #       - Solve for optimal values
    #       - Extract policy
    #       - Save results
    print("\nRunning Value Iteration...")
    vi_solver = ValueIteration(env, gamma=args.gamma, epsilon=args.epsilon)
    vi_values, vi_iterations, vi_history = vi_solver.solve(max_iterations=args.max_iter)
    vi_policy =  vi_solver.extract_policy(vi_values)
    np.savez('results/value_function.npz', 
             values=vi_values, 
             policy=vi_policy, 
             iterations=vi_iterations,
             history=vi_history)
    
    print(f"Value Iteration converged in {vi_iterations} iterations.")

    # TODO: Run Q-Iteration
    #       - Create QIteration solver
    #       - Solve for optimal Q-values
    #       - Extract policy and values
    #       - Save results
    print("\nRunning Q-Iteration...")
    qi_solver = QIteration(env, gamma=args.gamma, epsilon=args.epsilon)
    qi_q_values, qi_iterations, qi_history = qi_solver.solve(max_iterations=args.max_iter)
    qi_policy = qi_solver.extract_policy(qi_q_values)
    qi_values = qi_solver.extract_values(qi_q_values)
    np.savez('results/q_value_function.npz', 
             q_values=qi_q_values, 
             values=qi_values,
             policy=qi_policy, 
             iterations=qi_iterations,
             history=qi_history)
    
    print(f"Q-Iteration converged in {qi_iterations} iterations.")
    # TODO: Compare algorithms
    #       - Print convergence statistics
    #       - Check if policies match
    #       - Save comparison results
    print("\nComparing policies...")

    policies_match = np.array_equal(vi_policy, qi_policy)
    match_percent = np.mean(vi_policy == qi_policy) * 100
    print(f"Policy match: {match_percent:.2f}%")

    max_valu_diff = np.max(np.abs(vi_values - qi_values))
    print(f"Maximum value function difference: {max_valu_diff:.6f}")

    comparison_results = {
        'vi_iterations': int(vi_iterations),
        'qi_iterations': int(qi_iterations),
        'policies_match': bool(policies_match),
        'policy_match_percent': float(match_percent),
        'max_value_difference': float(max_valu_diff)
    }

    with open('results/comparison_results.json', 'w') as f:
        json.dump(comparison_results, f, indent=4)

    print("\nTraining complete. Results saved in 'results/' directory.")


if __name__ == '__main__':
    main()