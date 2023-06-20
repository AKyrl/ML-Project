from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf
from matplotlib import pyplot as plt
from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import policy_gradient
from open_spiel.python.algorithms import nfsp
from open_spiel.python.algorithms import cfr
import pyspiel
import kuhn_utils as pu

FLAGS = flags.FLAGS

flags.DEFINE_integer("eval_every", 10000,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_list("hidden_layers_sizes", [
  128,
], "Number of hidden units in the avg-net and Q-net.")
flags.DEFINE_integer("replay_buffer_capacity", int(2e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e6),
                     "Size of the reservoir buffer.")
flags.DEFINE_float("anticipatory_param", 0.1,
                   "Prob of using the rl best response as episode policy.")
flags.DEFINE_enum("loss_str", "rpg", ["a2c", "rpg", "qpg", "rm"],
                  "PG loss to use.")
flags.DEFINE_integer("num_episodes", int(1e6), "Number of train episodes.")

PG_PATH = "../policies/PG_Kuhn"
NFSP_PATH = "../policies/NFSP_Kuhn"
CFR_PATH = "../policies/CFR_Kuhn"


def play_against_random(game, policies, episodes, iterations):
  for ep in range(episodes):
    utility_res = {algo: 0 for algo in list(policies.keys())}
    money_won = {algo: 0 for algo in list(policies.keys())}
    player_order = list(policies.keys())
    for i in range(iterations):
      state = game.new_initial_state()
      cards = []
      agent_decisions = []
      # We change the order of players after half to account for who plays first.
      if i == iterations / 2 - 1:
        player_order[0], player_order[1] = player_order[1], player_order[0]
      while not state.is_terminal():
        if state.is_chance_node():
          outcomes = state.chance_outcomes()
          card_list, prob_list = zip(*outcomes)
          card = np.random.choice(card_list, p=prob_list)
          cards.append(state.action_to_string(card))
          state.apply_action(card)
        else:
          used_policy = player_order[state.current_player()]
          agent_decision = np.random.choice(state.legal_actions(state.current_player()),
                                      p=list(policies[used_policy].action_probabilities(state).values()))
          agent_decisions.append(state.action_to_string(agent_decision))
          state.apply_action(agent_decision)

      utility_res[player_order[0]] += state.returns()[0]
      utility_res[player_order[1]] += state.returns()[1]
      money_won[player_order[0]] = money_won[player_order[0]] + 1 if state.returns()[0] > state.returns()[1] else money_won[player_order[0]]
      money_won[player_order[1]] = money_won[player_order[1]] + 1 if state.returns()[1] > state.returns()[0] else money_won[player_order[1]]

    print(f"{ep}\tWins: {str(money_won)}")
    print(f"\tUtility: {str(utility_res)}")
  return utility_res

def train_nfsp(game):
  num_players = 2

  env_configs = {"players": num_players}
  env = rl_environment.Environment(game, **env_configs)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]
  game = pyspiel.load_game(game)
  hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
  kwargs = {
    "replay_buffer_capacity": FLAGS.replay_buffer_capacity,
    "epsilon_decay_duration": FLAGS.num_train_episodes,
    "epsilon_start": 0.06,
    "epsilon_end": 0.001,
  }

  with tf.Session() as sess:
    agents = [
      nfsp.NFSP(sess, idx, info_state_size, num_actions, hidden_layers_sizes,
                FLAGS.reservoir_buffer_capacity, FLAGS.anticipatory_param,
                **kwargs) for idx in range(num_players)

    ]
    expl_policies_avg = pu.NFSP(env, agents, nfsp.MODE.average_policy)

    sess.run(tf.global_variables_initializer())
    for ep in range(FLAGS.num_episodes):
      if (ep + 1) % FLAGS.eval_every == 0:
        losses = [agent.loss for agent in agents]
        logging.info("Losses: %s", losses)
        expl = exploitability.exploitability(env.game, expl_policies_avg)
        logging.info("[%s] Exploitability AVG %s", ep + 1, expl)
        logging.info("_____________________________________________")
        save_policy_to_path(game, expl_policies_avg, "policies/NFSP_Kuhn/{}".format(ep + 1))

      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = agents[player_id].step(time_step)
        action_list = [agent_output.action]
        time_step = env.step(action_list)

      for agent in agents:
        agent.step(time_step)
    save_policy_to_path(game, expl_policies_avg, "policies/NFSP_Kuhn/{}".format(ep+1))

def train_pg(game):
  num_players = 2

  env_configs = {"players": num_players}
  env = rl_environment.Environment(game, **env_configs)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]
  game = pyspiel.load_game(game)
  with tf.Session() as sess:
    agents = [
      policy_gradient.PolicyGradient(
        sess,
        idx,
        info_state_size,
        num_actions,
        loss_str=FLAGS.loss_str,
        hidden_layers_sizes=(128,)) for idx in range(num_players)
    ]
    expl_policies_avg = pu.PG(env, agents)

    sess.run(tf.global_variables_initializer())

    for ep in range(FLAGS.num_episodes):
      if (ep + 1) % FLAGS.eval_every == 0:
        losses = [agent.loss for agent in agents]
        expl = exploitability.exploitability(env.game, expl_policies_avg)
        msg = "-" * 80 + "\n"
        msg += f"{ep+1}: {expl}\n{losses}\n"
        logging.info("%s", msg)
        save_policy_to_path(game, expl_policies_avg, "policies/PG_Kuhn/{}".format(ep+1))

      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = agents[player_id].step(time_step)
        action_list = [agent_output.action]
        time_step = env.step(action_list)

      for agent in agents:
        agent.step(time_step)

    save_policy_to_path(game, expl_policies_avg, f"policies/PG_Kuhn/{ep}")

def save_policy_to_path(game, expl_policies_avg, path):
  tabular_policy = policy.tabular_policy_from_callable(game, expl_policies_avg)
  pu.save_tabular_policy(game, tabular_policy, path)

def train_cfr(game):
  game = pyspiel.load_game(game)
  cfr_solver = cfr.CFRSolver(game)
  for iter in range(FLAGS.num_episodes):
    if (iter + 1) % FLAGS.eval_every == 0:
      save_cfr_policy(game, cfr_solver, iter + 1)
      print(f"Iteration {iter} - saved policy")
    cfr_solver.evaluate_and_update_policy()
  save_cfr_policy(game, cfr_solver, FLAGS.num_episodes)

def save_cfr_policy(game, cfr_solver, iteration):
  average_policy = cfr_solver.average_policy()
  policy = dict(zip(average_policy.state_lookup, average_policy.action_probability_array))
  pu.save_policy(game, policy, f"policies/CFR_Kuhn/{iteration}")

def plot_policy(path, game_name, algo_name):
  game = pyspiel.load_game(game_name)

  files = np.array([])
  for (root, subFolder, filenames) in os.walk(path):
    for file in filenames:
      p = os.path.join(root, file)
      files = np.append(files, p)
  algo_policies = {}

  for file in files:
    algo_iterations = int(file.split("/")[2])
    algo_policy = pu.load_to_tabular_policy(file)
    algo_policies[algo_iterations] = policy.tabular_policy_from_callable(game, algo_policy)

  algo_exploitabilities = {}
  algo_nashconvs = {}
  for key in algo_policies:
    algo_exploitabilities[key] = exploitability.exploitability(game, algo_policies[key])
    algo_nashconvs[key] = exploitability.nash_conv(game, algo_policies[key])

  plot_results('Exploitability - training iterations', 'Exploitability', algo_exploitabilities, algo_name, game_name)
  plot_results('NashConv - training iterations', 'NashConv', algo_nashconvs, algo_name, game_name)

legend = []

def plot_results(title, metric, results, algorithm, game_name):
  sorted_dict = dict(sorted(results.items()))
  n_iterations = sorted_dict.keys()
  values = sorted_dict.values()

  plt.plot(n_iterations, values)
  legend.append(algorithm)
  plt.title(title)
  plt.legend(legend)
  plt.xlabel('Number of training iterations')
  plt.ylabel(metric)
  plt.yscale('log')
  graph_name = f'img/{game_name}_{algorithm}_{metric}'
  plt.savefig(graph_name)
  plt.clf()

def play_kuhn(game):
  game = pyspiel.load_game(game)
  random = pu.load_to_tabular_policy('../policies/NFSP_Kuhn/10000')
  policy = pu.load_to_tabular_policy('../policies/CFR_Kuhn/1000000')
  play_against_random(game, {'CFR': policy, 'Random': random}, 10, int(1e4))

def main(_):
  game = "kuhn_poker"
  train_nfsp(game)
  train_pg(game)
  train_cfr(game)
  plot_policy(PG_PATH, game, PG_PATH.split("/")[1][:2])
  plot_policy(NFSP_PATH, game, NFSP_PATH.split("/")[1][:4])
  plot_policy(CFR_PATH, game, CFR_PATH.split("/")[1][:3])
  play_kuhn(game)


if __name__ == "__main__":
  app.run(main)
