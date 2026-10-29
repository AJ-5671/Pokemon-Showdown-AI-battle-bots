import asyncio
from poke_env.player import RandomPlayer

async def main():
    player1 = RandomPlayer(battle_format="gen8randombattle")
    player2 = RandomPlayer(battle_format="gen8randombattle")
    await player1.battle_against(player2, n_battles=1)
    print(f"Player 1 won {player1.n_won_battles} / 1 battles")

asyncio.run(main())
