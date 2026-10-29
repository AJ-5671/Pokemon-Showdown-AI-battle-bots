import asyncio
import matplotlib.pyplot as plt
from poke_env.player import RandomPlayer, MaxBasePowerPlayer
from dqn_agent import DQNAgent
from rl_player import RLPlayer, STATE_DIM, ACTION_DIM

# ── Config ────────────────────────────────────────────────────────
MODEL_PATH = "dqn_model.pth"
N_EVAL     = 100   # battles per evaluation
# ─────────────────────────────────────────────────────────────────


def print_results(label: str, won: int, total: int):
    rate = won / total * 100
    bar  = "█" * int(rate / 5) + "░" * (20 - int(rate / 5))
    print(f"\n  {label}")
    print(f"  [{bar}] {rate:.1f}%  ({won}/{total} wins)")


async def evaluate_vs(agent: DQNAgent, opponent, label: str, n: int = N_EVAL):
    """Run n battles against a given opponent and return win rate."""
    agent.epsilon = 0.0   # pure exploitation — no random actions

    player = RLPlayer(
        agent          = agent,
        training       = False,
        battle_format  = "gen8randombattle"
    )

    await player.battle_against(opponent, n_battles=n)

    won      = player.n_won_battles
    win_rate = won / n
    print_results(label, won, n)
    return win_rate


def plot_results(labels: list, win_rates: list):
    """Bar chart comparing win rates against different opponents."""
    colors = ["steelblue", "tomato", "gray"]
    bars   = plt.bar(labels, [r * 100 for r in win_rates],
                     color=colors[:len(labels)], edgecolor="black", width=0.4)

    # Value labels on bars
    for bar, rate in zip(bars, win_rates):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{rate*100:.1f}%",
            ha="center", fontsize=12, fontweight="bold"
        )

    plt.axhline(y=50, color="green", linestyle="--", linewidth=1.2,
                label="Random baseline (50%)")
    plt.ylim(0, 105)
    plt.ylabel("Win Rate (%)")
    plt.title("DQN Agent — Evaluation Results")
    plt.legend()
    plt.tight_layout()
    plt.savefig("evaluation_results.png")
    plt.show()
    print("\nEvaluation plot saved to evaluation_results.png")


async def main():
    print("=" * 50)
    print("  Pokémon Showdown — DQN Agent Evaluation")
    print("=" * 50)
    print(f"\n  Loading model from {MODEL_PATH}...")

    # Load trained agent
    agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    agent.load(MODEL_PATH)
    print("  Model loaded successfully!\n")

    labels    = []
    win_rates = []

    # ── Test 1: vs RandomPlayer ───────────────────────────────────
    print("─" * 50)
    print("  Test 1: DQN Agent vs Random Player")
    print("─" * 50)
    random_opp  = RandomPlayer(battle_format="gen8randombattle")
    wr_random   = await evaluate_vs(agent, random_opp, "vs RandomPlayer", N_EVAL)
    labels.append("vs Random")
    win_rates.append(wr_random)

    # ── Test 2: vs MaxDamagePlayer ────────────────────────────────
    print("\n" + "─" * 50)
    print("  Test 2: DQN Agent vs Max Damage Player")
    print("  (always picks the highest base-power move)")
    print("─" * 50)
    maxdmg_opp  = MaxBasePowerPlayer(battle_format="gen8randombattle")
    wr_maxdmg   = await evaluate_vs(agent, maxdmg_opp, "vs MaxDamagePlayer", N_EVAL)
    labels.append("vs MaxDamage")
    win_rates.append(wr_maxdmg)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  SUMMARY")
    print("=" * 50)
    for label, wr in zip(labels, win_rates):
        status = "✅ Beats baseline" if wr > 0.5 else "❌ Below baseline"
        print(f"  {label:<20} {wr*100:.1f}%   {status}")

    # ── Plot ──────────────────────────────────────────────────────
    plot_results(labels, win_rates)


if __name__ == "__main__":
    asyncio.run(main())