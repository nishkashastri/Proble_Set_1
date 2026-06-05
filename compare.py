import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

from vi import value_iteration
from pi import policy_iteration


def vi_total_work(k, nS, nA):
    return k * nS * nA * nS


def pi_total_work(k, nS):
    return k * (nS ** 3)


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=True)
    P  = env.unwrapped.P
    nS = env.observation_space.n
    nA = env.action_space.n

    gammas = [0.5, 0.9, 0.99, 0.999]
    theta  = 1e-4

    vi_iters, vi_times, vi_work_list = [], [], []
    pi_iters, pi_times, pi_work_list = [], [], []

    hdr = (f"{'gamma':>6} | {'VI iters':>10} | {'VI time (s)':>12} | {'VI work':>12}"
           f" | {'PI iters':>10} | {'PI time (s)':>12} | {'PI work':>12}")
    print(hdr)
    print("-" * len(hdr))

    for gamma in gammas:
        t0 = time.perf_counter()
        _, _, k_vi = value_iteration(P, gamma, theta)
        t_vi = time.perf_counter() - t0

        t0 = time.perf_counter()
        _, _, k_pi = policy_iteration(P, gamma)
        t_pi = time.perf_counter() - t0

        w_vi = vi_total_work(k_vi, nS, nA)
        w_pi = pi_total_work(k_pi, nS)

        vi_iters.append(k_vi); vi_times.append(t_vi); vi_work_list.append(w_vi)
        pi_iters.append(k_pi); pi_times.append(t_pi); pi_work_list.append(w_pi)

        print(f"{gamma:>6} | {k_vi:>10} | {t_vi:>12.6f} | {w_vi:>12}"
              f" | {k_pi:>10} | {t_pi:>12.6f} | {w_pi:>12}")

    env.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(gammas, vi_iters, 'b-o', label='Value Iteration',  lw=2, ms=8)
    ax.plot(gammas, pi_iters, 'r-s', label='Policy Iteration', lw=2, ms=8)
    ax.set_xlabel('Discount Factor (gamma)', fontsize=12)
    ax.set_ylabel('Iteration Count',         fontsize=12)
    ax.set_title('Iteration Count vs. gamma  (FrozenLake-v1, is_slippery=True)', fontsize=12)
    ax.legend(fontsize=11)
    ax.set_xticks(gammas)
    ax.set_xticklabels([str(g) for g in gammas])
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig('comparison_plot.png', dpi=150)
    print("\nPlot saved to comparison_plot.png")
    plt.show()
