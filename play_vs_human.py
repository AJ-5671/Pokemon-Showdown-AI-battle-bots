import asyncio
from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from dqn_agent import DQNAgent
from rl_player import RLPlayer, STATE_DIM, ACTION_DIM
from heuristic_agent import HeuristicAgent
from minimax_agent import MinimaxAgent


async def main():
    print("=" * 50)
    print("  Play vs AI — Pokémon Showdown")
    print("=" * 50)
    print("\nChoose your opponent:")
    print("  1. Random Agent    (easiest)")
    print("  2. Heuristic Agent (medium)")
    print("  3. DQN Agent       (medium)")
    print("  4. Minimax Agent   (hardest)")

    choice = input("\nEnter 1-4: ").strip()

    if choice == "1":
        ai = RandomPlayer(
            battle_format="gen8randombattle",
            account_configuration=AccountConfiguration("RandomAgent", None)
        )
        ai_name = "RandomAgent"

    elif choice == "2":
        ai = HeuristicAgent(
            battle_format="gen8randombattle",
            account_configuration=AccountConfiguration("HeuristicAgent", None)
        )
        ai_name = "HeuristicAgent"

    elif choice == "3":
        dqn = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
        dqn.load("dqn_model.pth")
        dqn.epsilon = 0.0
        ai = RLPlayer(
            agent=dqn,
            training=False,
            battle_format="gen8randombattle",
            account_configuration=AccountConfiguration("DQNAgent", None)
        )
        ai_name = "DQNAgent"

    elif choice == "4":
        ai = MinimaxAgent(
            battle_format="gen8randombattle",
            account_configuration=AccountConfiguration("MinimaxAgent", None)
        )
        ai_name = "MinimaxAgent"

    else:
        print("Invalid choice!")
        return

    print(f"\n✅ {ai_name} is ready and waiting for a challenge!")
    print("\n📋 How to play:")
    print("  1. Open http://localhost:8000 in your browser")
    print("  2. Log in with ANY username (e.g. 'Aditya')")
    print(f"  3. Challenge the agent — it will be logged in as: DQNAgent (or similar)")
    print("  5. Select 'Gen 8 Random Battle' format")
    print("  6. Click Challenge and start playing!")

    opponent_username = input("\nEnter YOUR Showdown username exactly: ").strip()
    print(f"\n⏳ Waiting for challenge from {opponent_username}...\n")

    await ai.accept_challenges(opponent_username, 1)

    print("\n" + "=" * 50)
    if ai.n_won_battles:
        print("  💀 The AI won! Better luck next time!")
    else:
        print("  🎉 YOU WIN! You beat the AI!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())