import numpy as np
from poke_env.player import Player
from poke_env.battle import AbstractBattle
from dqn_agent import DQNAgent

# Dimensions
STATE_DIM  = 14   # upgraded from 10
ACTION_DIM = 9

def embed_battle(battle: AbstractBattle) -> np.ndarray:
    """
    Enhanced state vector — now includes move base power.

    Vector layout (14 values):
    [0]    Our active Pokemon HP ratio
    [1]    Opponent active Pokemon HP ratio
    [2-5]  Type matchup multipliers for 4 moves  (normalized)
    [6]    Remaining friendly Pokemon             (normalized)
    [7]    Remaining opponent Pokemon             (normalized)
    [8]    Our active Pokemon speed               (normalized)
    [9]    Opponent active Pokemon speed          (normalized)
    [10-13] Base power of our 4 moves             (normalized)
    """
    state = np.zeros(STATE_DIM, dtype=np.float32)

    # [0] Our HP ratio
    if battle.active_pokemon:
        state[0] = battle.active_pokemon.current_hp_fraction

    # [1] Opponent HP ratio
    if battle.opponent_active_pokemon:
        state[1] = battle.opponent_active_pokemon.current_hp_fraction

    # [2-5] Type matchup + [10-13] base power for each move
    for i, move in enumerate(battle.available_moves[:4]):
        if battle.opponent_active_pokemon:
            multiplier   = battle.opponent_active_pokemon.damage_multiplier(move)
            state[2 + i] = multiplier / 4.0
        # Base power normalized (max ~250)
        state[10 + i] = (move.base_power or 0) / 250.0

    # [6] Remaining friendly Pokemon
    remaining_ours = sum(1 for p in battle.team.values() if not p.fainted)
    state[6] = remaining_ours / 6.0

    # [7] Remaining opponent Pokemon
    remaining_opp = sum(1 for p in battle.opponent_team.values() if not p.fainted)
    state[7] = remaining_opp / 6.0

    # [8] Our speed
    if battle.active_pokemon and battle.active_pokemon.base_stats:
        state[8] = battle.active_pokemon.base_stats.get("spe", 0) / 200.0

    # [9] Opponent speed
    if battle.opponent_active_pokemon and battle.opponent_active_pokemon.base_stats:
        state[9] = battle.opponent_active_pokemon.base_stats.get("spe", 0) / 200.0

    return state


def get_available_actions(battle: AbstractBattle) -> list:
    """
    Returns indices of all legal actions in the current battle state.
    Actions 0-3 → moves, Actions 4-8 → switches
    """
    actions = []

    for i in range(len(battle.available_moves)):
        actions.append(i)

    for i in range(len(battle.available_switches)):
        actions.append(4 + i)

    if not actions:
        actions = [0]

    return actions


class RLPlayer(Player):
    """
    A poke-env Player that uses the DQN agent to select actions.
    Handles the full RL loop: observe → act → reward → store → train.
    """

    def __init__(self, agent: DQNAgent, training: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.agent          = agent
        self.training       = training
        self.current_state  = None
        self.last_action    = None
        self.episode_reward = 0.0
        self.episode_rewards = []

    def choose_move(self, battle: AbstractBattle):
        """
        Called by poke-env every time the agent must make a decision.
        Observe → intermediate reward → act.
        """
        damaging = [m for m in battle.available_moves
                    if m.base_power and m.base_power > 0]
        if damaging:
            battle._available_moves = damaging

        state = embed_battle(battle)
        available = get_available_actions(battle)
        action = self.agent.select_action(state, available)

        # Store intermediate transition from previous step
        if self.training and self.current_state is not None:
            reward = self._compute_intermediate_reward(battle)
            self.episode_reward += reward
            self.agent.store(
                self.current_state,
                self.last_action,
                reward,
                state,
                done=False
            )
            self.agent.train_step()

        self.current_state = state
        self.last_action   = action

        return self._action_to_move(action, battle)

    def _compute_intermediate_reward(self, battle: AbstractBattle) -> float:
        reward = 0.0

        # Reward for damaging opponent
        if battle.opponent_active_pokemon:
            opp_hp = battle.opponent_active_pokemon.current_hp_fraction
            reward += (1.0 - opp_hp) * 0.1

        # Penalty for our pokemon taking damage
        if battle.active_pokemon:
            our_hp = battle.active_pokemon.current_hp_fraction
            reward -= (1.0 - our_hp) * 0.1

        # Reward for knocking out opponent pokemon
        fainted_opp = sum(1 for p in battle.opponent_team.values() if p.fainted)
        reward += fainted_opp * 0.5

        # Penalty for our pokemon fainting
        fainted_ours = sum(1 for p in battle.team.values() if p.fainted)
        reward -= fainted_ours * 0.5

        # Penalty for using zero-power moves (discourages stalling)
        if self.last_action is not None:
            moves = battle.available_moves
            if (self.last_action < len(moves) and
                    (moves[self.last_action].base_power or 0) == 0):
                reward -= 0.3

        # ── ADD THESE TWO BELOW ───────────────────────────────────────

        # Penalty for switching (discourages switch loops)
        if self.last_action is not None and self.last_action >= 4:
            reward -= 0.4

        # Bonus for keeping opponent low HP (encourages offensive play)
        if battle.opponent_active_pokemon:
            opp_hp = battle.opponent_active_pokemon.current_hp_fraction
            if opp_hp < 0.5:
                reward += 0.2

        return reward

    def _action_to_move(self, action: int, battle: AbstractBattle):
        """
        Converts an integer action index into a poke-env move order.
        Safely handles forced switches and edge cases.
        """
        available_moves    = battle.available_moves
        available_switches = battle.available_switches

        # Forced switch — must switch, no moves available
        if not available_moves and available_switches:
            switch_index = action - 4
            if 0 <= switch_index < len(available_switches):
                return self.create_order(available_switches[switch_index])
            return self.create_order(available_switches[0])

        # Normal move action (0-3)
        if action < 4:
            if action < len(available_moves):
                return self.create_order(available_moves[action])
            if available_moves:
                return self.create_order(available_moves[0])

        # Switch action (4-8)
        switch_index = action - 4
        if 0 <= switch_index < len(available_switches):
            return self.create_order(available_switches[switch_index])

        # Final fallback
        return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle: AbstractBattle):
        """
        Called when a battle ends.
        Assigns terminal reward and runs a final training step.
        """
        if not self.training or self.current_state is None:
            return

        # Terminal reward: +1 win, -1 loss
        reward = 1.0 if battle.won else -1.0
        self.episode_reward += reward

        next_state = np.zeros(STATE_DIM, dtype=np.float32)

        self.agent.store(
            self.current_state,
            self.last_action,
            reward,
            next_state,
            done=True
        )

        self.agent.train_step()
        self.agent.decay_epsilon()

        self.episode_rewards.append(self.episode_reward)
        self.episode_reward = 0.0
        self.current_state  = None
        self.last_action    = None