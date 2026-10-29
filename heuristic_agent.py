from poke_env.player import Player
from poke_env.battle import AbstractBattle


class HeuristicAgent(Player):
    """
    Rule-based agent using domain knowledge.

    Decision priority:
    1. If active Pokemon is low HP → switch to healthiest Pokemon
    2. If a move has type advantage (2x or 4x) → use it
    3. If opponent is faster and we have a super effective move → use it
    4. Otherwise → use highest base power move available
    """

    # HP threshold below which agent considers switching
    LOW_HP_THRESHOLD = 0.30

    def choose_move(self, battle: AbstractBattle):

        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        # ── Rule 0: Anti-stall — always prefer damaging moves ────────
        damaging_moves = [
            m for m in battle.available_moves
            if m.base_power and m.base_power > 0
        ]
        if not damaging_moves and not battle.available_switches:
            return self.choose_random_move(battle)

        # ── Rule 1: Low HP → switch to healthiest available Pokemon ──
        if (active and
                active.current_hp_fraction < self.LOW_HP_THRESHOLD and
                battle.available_switches):
            best_switch = max(
                battle.available_switches,
                key=lambda p: p.current_hp_fraction
            )
            return self.create_order(best_switch)

        # ── Rule 2: Use super effective move if available ─────────────
        if opponent and battle.available_moves:
            super_effective = [
                move for move in battle.available_moves
                if opponent.damage_multiplier(move) >= 2.0
            ]
            if super_effective:
                # Among super effective moves, pick highest base power
                best_move = max(
                    super_effective,
                    key=lambda m: m.base_power or 0
                )
                return self.create_order(best_move)

        # ── Rule 3: If opponent faster + we have no advantage → switch ─
        if (active and opponent and battle.available_switches):
            our_speed = active.base_stats.get("spe", 0)
            opp_speed = opponent.base_stats.get("spe", 0)

            if opp_speed > our_speed * 1.2:  # opponent is 20% faster
                # Check if any switch has a type advantage
                for switch in battle.available_switches:
                    for move in battle.available_moves:
                        if opponent.damage_multiplier(move) >= 2.0:
                            break
                    else:
                        # No advantage here, try switching
                        best_switch = max(
                            battle.available_switches,
                            key=lambda p: p.current_hp_fraction
                        )
                        return self.create_order(best_switch)

        # ── Rule 4: Fallback → highest base power move ────────────────
        if battle.available_moves:
            best_move = max(
                battle.available_moves,
                key=lambda m: m.base_power or 0
            )
            return self.create_order(best_move)

        # ── Final fallback ─────────────────────────────────────────────
        return self.choose_random_move(battle)



### 🧠 Decision Tree
'''
Is
HP < 30 %?
YES → Switch
to
healthiest
Pokemon
NO  ↓
Is
there
a
super
effective
move?
YES → Use
highest
power
super
effective
move
NO  ↓
Is
opponent
20 % faster
than
us?
YES → Try
switching
NO  ↓
Use
highest
base
power
move
available'''