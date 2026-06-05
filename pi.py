import gymnasium as gym
import numpy as np

from vi import (print_value_grid, print_policy_arrows,
                value_iteration, ARROWS_ASCII)


def policy_iteration(P, gamma):
    nS = len(P)
    nA = len(P[0])
    policy = np.zeros(nS, dtype=int)
    k = 0

    while True:
        P_pi = np.zeros((nS, nS))
        R_pi = np.zeros(nS)
        for s in range(nS):
            a = policy[s]
            for prob, s2, r, _ in P[s][a]:
                P_pi[s, s2] += prob
                R_pi[s]     += prob * r

        V = np.linalg.solve(np.eye(nS) - gamma * P_pi, R_pi)

        new_policy = np.zeros(nS, dtype=int)
        for s in range(nS):
            q = np.array([
                sum(p * (r + gamma * V[s2]) for p, s2, r, _ in P[s][a])
                for a in range(nA)
            ])
            # keep current action on tie — prevents cycling from numerical near-ties
            if q[policy[s]] >= q.max() - 1e-10:
                new_policy[s] = policy[s]
            else:
                new_policy[s] = int(q.argmax())

        k += 1
        if np.array_equal(new_policy, policy):
            break
        policy = new_policy

    return V, policy, k


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=True)
    P = env.unwrapped.P

    gamma = 0.99

    print("=" * 60)
    print(f"Policy Iteration  (gamma={gamma})")
    print("=" * 60)

    V_pi, policy_pi, k_pi = policy_iteration(P, gamma)

    print(f"Iterations to converge: {k_pi}")
    print_value_grid(V_pi, "V*")
    print_policy_arrows(policy_pi, "pi*")

    print("\n--- Comparison with Value Iteration ---")
    V_vi, policy_vi, k_vi = value_iteration(P, gamma, theta=1e-4)

    if np.array_equal(policy_pi, policy_vi):
        print("VI and PI return the SAME policy.")
    else:
        print("VI and PI return DIFFERENT policies.")
        diff = np.where(policy_pi != policy_vi)[0]
        print(f"  Differing states: {diff.tolist()}")
        for s in diff:
            print(f"  State {s:2d}: PI={ARROWS_ASCII[policy_pi[s]]}  VI={ARROWS_ASCII[policy_vi[s]]}")
        print("  (Ties in argmax can break differently — both are optimal.)")

    print(f"\n  Max |V*_PI - V*_VI| = {np.max(np.abs(V_pi - V_vi)):.2e}")

    env.close()
