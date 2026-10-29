from poke_env.player import Player
from poke_env.battle import AbstractBattle


class MinimaxAgent(Player):
    """
    Minimax agent with alpha-beta pruning.

    Looks ahead N turns and picks the action that leads to
    the best guaranteed outcome, assuming the opponent plays optimally.

    Corresponds to 'Minimax Agent' in the project synopsis.
    """

    DEPTH = 2  # how many turns ahead to search

    def choose_move(self, battle: AbstractBattle):
        best_score = float("-inf")
        best_action = None

        # Try every available action and score it
        for action in self._get_actions(battle):
            score = self._minimax(
                battle=battle,
                action=action,
                depth=self.DEPTH,
                alpha=float("-inf"),
                beta=float("inf"),
                is_maximizing=False  # opponent moves next
            )
            if score > best_score:
                best_score = score
                best_action = action

        if best_action is None:
            return self.choose_random_move(battle)

        return self._action_to_order(best_action, battle)

    # ── Minimax with Alpha-Beta Pruning ───────────────────────────────

    def _minimax(self, battle, action, depth, alpha, beta, is_maximizing):
        """
        Recursively evaluate future states.

        is_maximizing=True  → our turn  (maximize score)
        is_maximizing=False → opponent  (minimize score)
        """
        # Base case — evaluate current position
        if depth == 0:
            return self._evaluate(battle, action)

        if is_maximizing:
            max_score = float("-inf")
            for next_action in self._get_actions(battle):
                score = self._minimax(
                    battle=battle,
                    action=next_action,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    is_maximizing=False
                )
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break  # ✂️ Beta cutoff — prune this branch
            return max_score

        else:
            min_score = float("inf")
            for next_action in self._get_actions(battle):
                score = self._minimax(
                    battle=battle,
                    action=next_action,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    is_maximizing=True
                )
                min_score = min(min_score, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break  # ✂️ Alpha cutoff — prune this branch
            return min_score

    # ── Position Evaluation Function ──────────────────────────────────

    def _evaluate(self, battle: AbstractBattle, action) -> float:
        """
        Scores the current battle state from our perspective.
        Higher score = better position for us.

        Factors:
        - HP advantage
        - Pokemon remaining advantage
        - Type matchup of chosen action
        - Speed advantage
        """
        score = 0.0

        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        # ── HP advantage ──────────────────────────────────────────────
        if active:
            score += active.current_hp_fraction * 2.0
        if opponent:
            score -= opponent.current_hp_fraction * 2.0

        # ── Pokemon remaining advantage ───────────────────────────────
        our_remaining = sum(1 for p in battle.team.values()
                            if not p.fainted)
        opp_remaining = sum(1 for p in battle.opponent_team.values()
                            if not p.fainted)
        score += (our_remaining - opp_remaining) * 0.5

        # ── Type matchup of chosen action ─────────────────────────────
        if action < 4 and opponent:
            moves = battle.available_moves
            if action < len(moves):
                multiplier = opponent.damage_multiplier(moves[action])
                if multiplier >= 4.0:
                    score += 3.0  # super effective x4
                elif multiplier >= 2.0:
                    score += 1.5  # super effective x2
                elif multiplier <= 0.5:
                    score -= 1.5  # not very effective
                elif multiplier == 0:
                    score -= 3.0  # immune

        # ── Base power of chosen move ─────────────────────────────────
        if action < 4:
            moves = battle.available_moves
            if action < len(moves):
                score += (moves[action].base_power or 0) / 100.0

        # ── Speed advantage ───────────────────────────────────────────
        if active and opponent:
            our_speed = active.base_stats.get("spe", 0)
            opp_speed = opponent.base_stats.get("spe", 0)
            if our_speed > opp_speed:
                score += 0.5  # we move first
            else:
                score -= 0.5  # opponent moves first

        return score

    # ── Helper Methods ────────────────────────────────────────────────

    def _get_actions(self, battle: AbstractBattle) -> list:
        """Returns list of valid action indices (0-3 moves, 4-8 switches)."""
        actions = []
        for i in range(len(battle.available_moves)):
            actions.append(i)
        for i in range(len(battle.available_switches)):
            actions.append(4 + i)
        if not actions:
            actions = [0]
        return actions

    def _action_to_order(self, action: int, battle: AbstractBattle):
        """Converts action index to poke-env move order."""
        available_moves = battle.available_moves
        available_switches = battle.available_switches

        # Forced switch
        if not available_moves and available_switches:
            switch_index = action - 4
            if 0 <= switch_index < len(available_switches):
                return self.create_order(available_switches[switch_index])
            return self.create_order(available_switches[0])

        # Move (0-3)
        if action < 4:
            if action < len(available_moves):
                return self.create_order(available_moves[action])
            if available_moves:
                return self.create_order(available_moves[0])

        # Switch (4-8)
        switch_index = action - 4
        if 0 <= switch_index < len(available_switches):
            return self.create_order(available_switches[switch_index])

        return self.choose_random_move(battle)



'''
### 🧠 How Minimax Works — Simple Explanation
```
Our
turn — try every action:
    ├── Use
    Flamethrower
│   └── Opponent
's best response:
│       ├── Use
Water
Gun    → score = -1.0(bad for us)
│       └── Use
Tackle       → score = +0.5
│           └── Our
best
response...(depth - 1)
│       Opponent
picks
minimum → -1.0
│
├── Use
Thunderbolt
│   └── Opponent
's best response:
│       ├── Use
Earthquake   → score = +2.0
│       └── Use
Tackle       → score = +1.5
│       Opponent
picks
minimum → +1.5  ← BEST
for us!
│
└── Switch
to
Charizard
└── Opponent
's best response: → score = +0.3

We
pick
MAXIMUM → Use
Thunderbolt ✅
```

### ✂️ Alpha-Beta Pruning

Without
pruning: checks
ALL
branches(slow)
With
pruning: skips
branches
that
can
't possibly be better (fast)

Example: If
we
already
found
a
move
scoring + 2.0,
and current
branch
scores - 1.0
for opponent...
    we
    SKIP
    the
    rest
    of
    that
    branch
    entirely!'''