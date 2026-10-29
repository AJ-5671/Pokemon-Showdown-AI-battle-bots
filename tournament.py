import asyncio
import itertools
import matplotlib.pyplot as plt
import numpy as np
from poke_env.player import RandomPlayer
from dqn_agent import DQNAgent
from rl_player import RLPlayer, STATE_DIM, ACTION_DIM
from heuristic_agent import HeuristicAgent
from minimax_agent import MinimaxAgent

# ── Config ────────────────────────────────────────────────────────
N_BATTLES  = 50    # battles per matchup
MODEL_PATH = "dqn_model.pth"
# ─────────────────────────────────────────────────────────────────


def print_header(text: str):
    print("\n" + "=" * 55)
    print(f"  {text}")
    print("=" * 55)


def print_matchup(p1: str, p2: str, p1_wins: int, p2_wins: int, total: int):
    p1_rate = p1_wins / total * 100
    p2_rate = p2_wins / total * 100
    print(f"\n  {p1:<20} vs  {p2:<20}")
    print(f"  {'█' * int(p1_rate/5):<20} {p1_rate:.1f}% ({p1_wins}W)")
    print(f"  {'█' * int(p2_rate/5):<20} {p2_rate:.1f}% ({p2_wins}W)")


def plot_results(agents: list, win_matrix: dict):
    """
    Two charts:
    1. Win rate heatmap (all vs all)
    2. Overall win rate bar chart
    """
    n      = len(agents)
    matrix = np.zeros((n, n))

    for i, a1 in enumerate(agents):
        for j, a2 in enumerate(agents):
            if i != j and (a1, a2) in win_matrix:
                matrix[i][j] = win_matrix[(a1, a2)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Heatmap ───────────────────────────────────────────────────
    im = ax1.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100)
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(agents, rotation=45, ha="right")
    ax1.set_yticklabels(agents)
    ax1.set_title("Win Rate Heatmap\n(row agent vs column agent)")
    plt.colorbar(im, ax=ax1, label="Win Rate %")

    for i in range(n):
        for j in range(n):
            if i != j:
                ax1.text(j, i, f"{matrix[i][j]:.0f}%",
                         ha="center", va="center",
                         color="black", fontsize=10, fontweight="bold")

    # ── Overall win rate bar chart ────────────────────────────────
    overall = []
    for i, agent in enumerate(agents):
        rates = [matrix[i][j] for j in range(n) if i != j]
        overall.append(np.mean(rates) if rates else 0)

    colors = ["gray", "steelblue", "orange", "green"]
    bars   = ax2.bar(agents, overall, color=colors, edgecolor="black", width=0.5)

    for bar, val in zip(bars, overall):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center", fontsize=11, fontweight="bold"
        )

    ax2.axhline(y=50, color="red", linestyle="--", linewidth=1.2,
                label="50% baseline")
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("Average Win Rate (%)")
    ax2.set_title("Overall Agent Performance")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("tournament_results.png")
    plt.show()
    print("\n  Tournament plot saved to tournament_results.png")


async def run_matchup(p1, p2, p1_name: str, p2_name: str, n: int):
    """Run n battles between two players and return win counts."""
    print(f"\n  ⚔️  {p1_name} vs {p2_name} ({n} battles)...")

    # Record wins BEFORE the matchup
    p1_wins_before = p1.n_won_battles
    p2_wins_before = p2.n_won_battles

    await p1.battle_against(p2, n_battles=n)

    # Delta wins = wins gained in THIS matchup only
    p1_wins = p1.n_won_battles - p1_wins_before
    p2_wins = p2.n_won_battles - p2_wins_before

    # Fallback in case poke-env doesn't track opponent wins
    if p1_wins + p2_wins != n:
        p2_wins = n - p1_wins

    print_matchup(p1_name, p2_name, p1_wins, p2_wins, n)
    return p1_wins, p2_wins


async def tournament():
    print_header("Pokémon Showdown — Agent Tournament")

    # ── Initialise all agents ─────────────────────────────────────
    print("\n  Loading agents...")

    # Random
    random_agent = RandomPlayer(battle_format="gen8randombattle")

    # Heuristic
    heuristic_agent = HeuristicAgent(battle_format="gen8randombattle")

    # Minimax
    minimax_agent = MinimaxAgent(battle_format="gen8randombattle")

    # DQN
    dqn = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    dqn.load(MODEL_PATH)
    dqn.epsilon = 0.0   # no exploration during tournament
    dqn_agent   = RLPlayer(
        agent          = dqn,
        training       = False,
        battle_format  = "gen8randombattle"
    )

    print("  All agents loaded! ✅")

    # ── Agent registry ────────────────────────────────────────────
    agent_names = ["Random", "Heuristic", "Minimax", "DQN"]
    agent_list  = [random_agent, heuristic_agent,
                   minimax_agent, dqn_agent]

    win_matrix  = {}   # (name1, name2) → win rate %

    # ── Round robin ───────────────────────────────────────────────
    print_header("Round Robin Matchups")

    for (i, p1), (j, p2) in itertools.combinations(enumerate(agent_list), 2):
        name1 = agent_names[i]
        name2 = agent_names[j]

        p1_wins, p2_wins = await run_matchup(
            p1, p2, name1, name2, N_BATTLES
        )

        win_matrix[(name1, name2)] = p1_wins / N_BATTLES * 100
        win_matrix[(name2, name1)] = p2_wins / N_BATTLES * 100

    # ── Final standings ───────────────────────────────────────────
    print_header("Final Standings")

    standings = []
    for name in agent_names:
        rates = [win_matrix[(name, other)]
                 for other in agent_names if other != name
                 and (name, other) in win_matrix]
        avg = np.mean(rates) if rates else 0
        standings.append((name, avg))

    standings.sort(key=lambda x: x[1], reverse=True)

    print()
    medals = ["🥇", "🥈", "🥉", "4️⃣ "]
    for rank, (name, avg) in enumerate(standings):
        bar = "█" * int(avg / 5)
        print(f"  {medals[rank]}  {name:<12} {bar:<20} {avg:.1f}%")

    # ── Plot ──────────────────────────────────────────────────────
    plot_results(agent_names, win_matrix)


if __name__ == "__main__":
    asyncio.run(tournament())
# ```
#
# ---
#
# ### What This Produces
# ```
# =======================================================
#   Pokémon Showdown — Agent Tournament
# =======================================================
#
#   Loading agents... ✅
#
# =======================================================
#   Round Robin Matchups
# =======================================================
#
#   ⚔️  Random vs Heuristic (50 battles)...
#   Random      ████░░░░░░░░░░░░░░░░  20.0% (10W)
#   Heuristic   ████████████████░░░░  80.0% (40W)
#
#   ⚔️  Random vs Minimax (50 battles)...
#   ...
#
# =======================================================
#   Final Standings
# =======================================================
#
#   🥇  Minimax      ████████████████████  82.0%
#   🥈  Heuristic    ████████████████░░░░  75.0%
#   🥉  DQN          ████████████░░░░░░░░  65.0%
#   4️⃣   Random       ████░░░░░░░░░░░░░░░░  20.0%