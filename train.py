import asyncio
import matplotlib.pyplot as plt
from poke_env.player import RandomPlayer
from dqn_agent import DQNAgent
from rl_player import RLPlayer, STATE_DIM, ACTION_DIM

# ── Hyperparameters ────────────────────────────────────────────────
N_BATTLES       = 1000   # total training battles
EVAL_INTERVAL   = 100    # evaluate win rate every N battles
TARGET_UPDATE   = 10     # sync target network every N battles
SAVE_PATH       = "dqn_model.pth"
# ──────────────────────────────────────────────────────────────────


def plot_training(episode_rewards: list, win_rates: list, eval_interval: int):
    """Plot training reward and win rate curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Episode rewards
    ax1.plot(episode_rewards, alpha=0.6, color="steelblue")
    ax1.set_title("Reward per Battle")
    ax1.set_xlabel("Battle")
    ax1.set_ylabel("Reward")

    # Win rate over time
    x = [(i + 1) * eval_interval for i in range(len(win_rates))]
    ax2.plot(x, win_rates, marker="o", color="green")
    ax2.set_title("Win Rate vs RandomPlayer")
    ax2.set_xlabel("Battle")
    ax2.set_ylabel("Win Rate")
    ax2.set_ylim(0, 1)
    ax2.axhline(y=0.5, color="red", linestyle="--", label="Random baseline (50%)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_results.png")
    plt.show()
    print("Training plot saved to training_results.png")


async def evaluate(agent: DQNAgent, n_eval: int = 20) -> float:
    """
    Run n_eval battles with epsilon=0 (pure exploitation)
    against RandomPlayer and return win rate.
    """
    saved_epsilon     = agent.epsilon
    agent.epsilon     = 0.0   # no exploration during eval

    eval_player       = RLPlayer(
        agent=agent,
        training=False,
        battle_format="gen8randombattle"
    )
    opponent          = RandomPlayer(battle_format="gen8randombattle")

    await eval_player.battle_against(opponent, n_battles=n_eval)

    win_rate          = eval_player.n_won_battles / n_eval
    agent.epsilon     = saved_epsilon   # restore epsilon
    return win_rate


async def train():
    print("=" * 50)
    print("  Pokémon Showdown — DQN Agent Training")
    print("=" * 50)

    # ── Initialise agent and players ──────────────────────────────
    agent = DQNAgent(
        state_dim      = STATE_DIM,
        action_dim     = ACTION_DIM,
        lr             = 1e-3,
        gamma          = 0.99,
        epsilon        = 1.0,
        epsilon_min    = 0.05,
        epsilon_decay  = 0.995,
        buffer_capacity= 10000,
        batch_size     = 64,
    )

    rl_player = RLPlayer(
        agent          = agent,
        training       = True,
        battle_format  = "gen8randombattle"
    )
    opponent  = RandomPlayer(battle_format="gen8randombattle")

    win_rates = []

    # ── Training loop ─────────────────────────────────────────────
    for i in range(0, N_BATTLES, EVAL_INTERVAL):
        battles_this_round = min(EVAL_INTERVAL, N_BATTLES - i)

        print(f"\n[Battles {i+1}–{i+battles_this_round}] "
              f"epsilon={agent.epsilon:.3f}  "
              f"memory={len(agent.memory)}")

        # Train for EVAL_INTERVAL battles
        await rl_player.battle_against(opponent, n_battles=battles_this_round)

        # Sync target network periodically
        if (i // EVAL_INTERVAL) % TARGET_UPDATE == 0:
            agent.update_target_network()
            print("  ↻  Target network updated")

        # Evaluate current policy
        win_rate = await evaluate(agent, n_eval=20)
        win_rates.append(win_rate)
        print(f"  ✓  Win rate: {win_rate*100:.1f}%")

    # ── Final save ────────────────────────────────────────────────
    agent.save(SAVE_PATH)
    print(f"\nTraining complete! Model saved to {SAVE_PATH}")

    # ── Plot results ──────────────────────────────────────────────
    plot_training(rl_player.episode_rewards, win_rates, EVAL_INTERVAL)


if __name__ == "__main__":
    asyncio.run(train())