from open_spiel.python import rl_environment
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python.algorithms import random_agent
from matrix_games_definitions import (biased_rps, dispersion, battle_of_the_sexes, subsidy)
from matrix_games_dynamics import plot_replicator_dynamics_2x2, plot_replicator_dynamics_3x3

import numpy as np

def play_ep(env, agents, is_evaluation=False):
  """Plays an episode and returns the agent outputs and action list."""
  
  time_step = env.reset()
  agent_outputs = [agent.step(time_step, is_evaluation) for agent in agents]
  action_list = [agent_output.action for agent_output in agent_outputs]
  time_step = env.step(action_list)  # progressing the environment
  for agent in agents:
    agent.step(time_step, is_evaluation)  # preparing agents for next episode AND/OR training
  return agent_outputs, action_list


def train(env, agents, episodes):
  """Trains agents by playing multiple episodes and returns the probabilities of each action 
  and the trained agents."""

  all_probs = []
  for _ in range(episodes):
    agent_outputs, action_list = play_ep(env, agents)
    all_probs.append([step_output.probs for step_output in agent_outputs])
  return all_probs, agents


def eval(env, agents, episodes):
  """Evaluates agents during multiple episodes and returns their returns."""

  returns = np.zeros(2)
  for _ in range(episodes):
    time_step = env.reset()
    agents_output = [agent.step(time_step, is_evaluation=True) for agent in agents]
    action_list = [agent_output.action for agent_output in agents_output]
    env.step(action_list)
    returns[0] += env.get_state.returns()[0]
    returns[1] += env.get_state.returns()[1]
  return returns


def play(game):
  """Creates game environment, trains agents, and evaluates them against each other
  and against a random agent."""


  print(f"Creating {game.get_type().long_name}...\n")
  all_probs = []
  for _ in range(6):
    env = rl_environment.Environment(game)
    num_players = env.num_players
    num_actions = env.action_spec()["num_actions"]
    agents = [
      tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
      for idx in range(num_players)
    ]

    probs, agents = train(env, agents, 10000)
    print(f"Probabilities:\n{all_probs[-10:-1]}")
    all_probs.append(probs)

    print("\nEvaluating 2 QLearner agents...")
    returns = eval(env, agents, 1000)
    print(f"Agent 0 returns: {returns[0]} - Agent 1 returns: {returns[1]}")

    print("\nEvaluating a QLearner agent VS a random agent...")
    returns = eval(env, [agents[0], random_agent.RandomAgent(1, num_actions, "Entropy Master 2000")], 1000)
    print(f"QLearner returns: {returns[0]} - Random Agent returns: {returns[1]}\n")
  return all_probs



def main():
  history = []
  history = play(biased_rps())
  plot_replicator_dynamics_3x3(biased_rps(), history)
  plot_replicator_dynamics_2x2(dispersion(), history)
  plot_replicator_dynamics_2x2(subsidy(), history)
  play(dispersion())
  play(battle_of_the_sexes())
  play(subsidy())


if __name__ == "__main__":
  main()

