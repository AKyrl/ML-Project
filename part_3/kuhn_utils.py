import pyspiel
from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.policy import TabularPolicy
import pickle

# ============== #
#   ALGORITHMS   #
# ============== #

class NFSP(policy.Policy):
  def __init__(self, env, nfsp_policies, mode):
    game = env.game
    player_ids = [0, 1]
    super(NFSP, self).__init__(game, player_ids)
    self._policies = nfsp_policies
    self._mode = mode
    self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
        state.information_state_tensor(cur_player))
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)

    with self._policies[cur_player].temp_mode_as(self._mode):
      p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions}
    return prob_dict


class PG(policy.Policy):
  def __init__(self, env, pg_policies):
    game = env.game
    player_ids = [0, 1]
    super(PG, self).__init__(game, player_ids)
    self._policies = pg_policies
    self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
        state.information_state_tensor(cur_player))
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)

    p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions}
    return prob_dict


# ======================= #
#   POLICY MANIPULATION   #
# ======================= #

def save_policy(game, policy, path):
    tabular_policy = TabularPolicy(game)
    state_lookup_order = list(tabular_policy.state_lookup.keys())
    policy = {k: policy.get(k) for k in state_lookup_order}
    tabular_policy.action_probability_array = list(list(val) for val in policy.values())
    save_tabular_policy(game, tabular_policy, path)

def save_tabular_policy(game, tabular_policy, path):
    dict = {'game': game.get_type().short_name, 'action_probability_array': tabular_policy.action_probability_array}
    with open(path, 'wb') as file:
        pickle.dump(dict, file)
    return tabular_policy

def load_to_tabular_policy(path):
    with open(path, 'rb') as file:
        dict = pickle.load(file)
        game = pyspiel.load_game(dict['game'])
        tabular_policy = TabularPolicy(game)
        tabular_policy.action_probability_array = dict['action_probability_array']
        return tabular_policy