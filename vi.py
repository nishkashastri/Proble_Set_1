import gymnasium as gym
import numpy as np


ARROWS_ASCII = ['<', 'v', '>', '^']
HOLES        = {5, 7, 11, 12}
GOAL         = 15


def value_iteration(P, gamma, theta):
    nS = len(P)
    nA = len(P[0])
    V  = np.zeros(nS)
    threshold = theta * (1 - gamma) / gamma
    k = 0

    while True:
        V_new = np.zeros(nS)
        for s in range(nS):
            q = np.array([
                sum(p * (r + gamma * V[s2]) for p, s2, r, _ in P[s][a])
                for a in range(nA)
            ])
            V_new[s] = q.max()

        converged = np.max(np.abs(V_new - V)) < threshold
        V = V_new
        k += 1
        if converged:
            break

    return V, _greedy(P, V, nS, nA, gamma), k


def value_iteration_with_history(P, gamma, theta):
    nS = len(P)
    nA = len(P[0])
    V  = np.zeros(nS)
    threshold = theta * (1 - gamma) / gamma
    k = 0
    V_hist, pi_hist = [], []

    while True:
        V_new = np.zeros(nS)
        for s in range(nS):
            q = np.array([
                sum(p * (r + gamma * V[s2]) for p, s2, r, _ in P[s][a])
                for a in range(nA)
            ])
            V_new[s] = q.max()

        converged = np.max(np.abs(V_new - V)) < threshold
        V = V_new
        k += 1
        V_hist.append(V.copy())
        pi_hist.append(_greedy(P, V, nS, nA, gamma))
        if converged:
            break

    return V, pi_hist[-1], k, V_hist, pi_hist


def _greedy(P, V, nS, nA, gamma):
    policy = np.zeros(nS, dtype=int)
    for s in range(nS):
        q = np.array([
            sum(p * (r + gamma * V[s2]) for p, s2, r, _ in P[s][a])
            for a in range(nA)
        ])
        policy[s] = q.argmax()
    return policy


def print_value_grid(V, title="V*"):
    print(f"\n{title}  (4x4 grid, row-major state order)")
    grid = V.reshape(4, 4)
    sep  = "+" + "----------+" * 4
    print(sep)
    for row in range(4):
        print("|", end="")
        for col in range(4):
            print(f"  {grid[row, col]:7.5f} |", end="")
        print()
        print(sep)


def print_policy_arrows(policy, title="pi*"):
    print(f"\n{title}")
    sep = "+" + "----+" * 4
    print(sep)
    for row in range(4):
        print("|", end="")
        for col in range(4):
            s = row * 4 + col
            if s in HOLES:
                sym = " H "
            elif s == GOAL:
                sym = " G "
            else:
                sym = f" {ARROWS_ASCII[policy[s]]} "
            print(f"{sym}|", end="")
        print()
        print(sep)


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=True)
    P = env.unwrapped.P

    gamma = 0.99
    theta = 1e-4

    print("=" * 60)
    print(f"Value Iteration  (gamma={gamma}, theta={theta})")
    print("=" * 60)

    V, policy, k, V_hist, pi_hist = value_iteration_with_history(P, gamma, theta)

    print(f"Iterations to converge: {k}")
    print_value_grid(V, "V*")
    print_policy_arrows(policy, "pi*")

    k_star = None
    for i, pi_k in enumerate(pi_hist):
        if np.array_equal(pi_k, policy):
            k_star = i + 1
            break

    if k_star is not None:
        err = np.max(np.abs(V_hist[k_star - 1] - V))
        print("\n--- Part (d): Policy Emergence ---")
        print(f"k* = {k_star}  (first iteration where greedy policy = optimal pi*)")
        print(f"||V_k* - V*||_inf = {err:.8f}")

    env.close()
