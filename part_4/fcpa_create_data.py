# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python spiel example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from threading import Thread

from absl import app
from absl import flags
import numpy as np
import pandas as pd

from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
import pyspiel
from random import randint, random
import re

import math as m
from fcpa_train import (chen, evaluate)
from fcpa_agent import get_agent_for_tournament, from_state_to_inputs

FLAGS = flags.FLAGS

global seed
seed = randint(0, 12761381)
flags.DEFINE_integer("seed", seed, "The seed to use for the RNG.")
# Supported types of players: "random", "human", "check_call", "fold"
flags.DEFINE_string("player0", "random", "Type of the agent for player 0.")
flags.DEFINE_string("player1", "random", "Type of the agent for player 1.")


def LoadAgent(agent_type, game, player_id, rng):
    """Return a bot based on the agent type."""
    if agent_type == "random":
        return uniform_random.UniformRandomBot(player_id, rng)
    elif agent_type == "human":
        return human.HumanBot()
    elif agent_type == "check_call":
        policy = pyspiel.PreferredActionPolicy([1, 0])
        return pyspiel.make_policy_bot(game, player_id, FLAGS.seed, policy)
    elif agent_type == "fold":
        policy = pyspiel.PreferredActionPolicy([0, 1])
        return pyspiel.make_policy_bot(game, player_id, FLAGS.seed, policy)
    elif agent_type == "NN":
        return get_agent_for_tournament(player_id)
    else:
        raise RuntimeError("Unrecognized agent type: {}".format(agent_type))


def run_game_once(agent1, agent2):
    rng = np.random.RandomState(randint(0, 12761381))
    games_list = pyspiel.registered_names()
    assert "universal_poker" in games_list

    fcpa_game_string = ("universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=150 100,"
                        "firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,"
                        "stack=20000 20000,bettingAbstraction=fcpa)")
    # print("Creating game: {}".format(fcpa_game_string))
    game = pyspiel.load_game(fcpa_game_string)

    agents = [
        LoadAgent(agent1, game, 0, rng),
        LoadAgent(agent2, game, 1, rng)
    ]

    state = game.new_initial_state()

    # print("INITIAL STATE")
    # print(str(state))

    data = []

    while not state.is_terminal():
        # The state can be three different types: chance node,
        # simultaneous node, or decision node
        current_player = state.current_player()
        if state.is_chance_node():
            # Chance node: sample an outcome
            outcomes = state.chance_outcomes()
            num_actions = len(outcomes)
            # print("Chance node with " + str(num_actions) + " outcomes")
            action_list, prob_list = zip(*outcomes)
            action = rng.choice(action_list, p=prob_list)
            # print("Sampled outcome: ",
            #       state.action_to_string(state.current_player(), action))
            state.apply_action(action)
        else:
            # Decision node: sample action for the single current player
            legal_actions = state.legal_actions()
            # for action in legal_actions:
            #     print("Legal action: {} ({})".format(
            #         state.action_to_string(current_player, action), action))
            action = agents[current_player].step(state)
            action_string = state.action_to_string(current_player, action)
            # print("Player ", current_player, ", chose action: ",
            #       action_string)
            round_data = from_state_to_inputs(state)
            round_data.append(action)
            round_data.append(current_player)
            data.append(round_data)
            state.apply_action(action)

        # print("")
        # print("NEXT STATE:")
        # print(str(state))

    returns = state.returns()
    # for pid in range(game.num_players()):
    #     print("Utility for player {} is {}".format(pid, returns[pid]))
    return data, returns


#
# def collect_data():
#     state, returns0, returns1 = run_game_once()
#     dataset = []
#     p0cards, p1cards, board_cards, rounds, acpc_state, action_sequence = parse_state_data(state)
#     chen0 = chen(p0cards)
#     chen1 = chen(p1cards)
#
#     spent_amounts = get_spent_amounts(action_sequence)
#     for round_idx in range(int(rounds)):
#         board_cards_curr_round = []
#         action0, action1 = get_last_action(round_idx, action_sequence)
#         if action0 == '' or action1 == '':
#             return dataset
#
#         if round_idx == 0:
#             pass
#         elif round_idx == 1:
#             board_cards_curr_round = fromStringToCardList(acpc_state[0])
#         elif round_idx == 2:
#             board_cards_curr_round = fromStringToCardList(acpc_state[0])
#             board_cards_curr_round.append(acpc_state[1])
#         elif round_idx == 3:
#             board_cards_curr_round = board_cards
#
#         curr_round = round_idx
#         spent0 = spent_amounts[round_idx][0]
#         spent1 = spent_amounts[round_idx][1]
#
#         # Player 0 data
#         # chen0, eval0, spent0, spent1, curr_round / action0, utility0
#         eval0 = evaluate(p0cards + board_cards_curr_round)
#         utility0 = returns0  # TODO: maybe change
#         player0data = [chen0, eval0, spent0, spent1, curr_round, action0, utility0]
#
#         # Player 1 data
#         # chen1, eval1, spent1, spent0, curr_round / action1, utility1
#         eval1 = evaluate(p1cards + board_cards_curr_round)
#         utility1 = returns1  # TODO: maybe change
#         player1data = [chen1, eval1, spent1, spent0, curr_round, action1, utility1]
#
#         print(f"p0: {player0data}")
#         print(f"p1: {player1data}")
#         dataset.append(player0data)
#         dataset.append(player1data)
#     return dataset
#
# def parse_action_sequence(action_sequence):
#     action_sequence = action_sequence[4:]
#
#     p = re.compile(r"(d{3})")
#     sep = p.sub(r',', action_sequence)
#     p = re.compile(r"(d{1})")
#     sep = p.sub(r',', sep).split(',')
#     return sep
#
# def get_spent_amounts(action_sequence):
#     # PREFLOP
#     counter = 1
#     player_money_round_0 = [150, 100]
#     for a in action_sequence[0]:
#         # Player 1 plays first (small blind), player 0 plays second (big blind)
#         # Every odd play is a play by player 1, every even play is a play by player 0
#         if counter % 2 != 0:
#             # player 1 plays
#             player_money_round_0[1] = check_action(a, player_money_round_0[1], player_money_round_0[0])
#         else:
#             # player 0 plays
#             player_money_round_0[0] = check_action(a, player_money_round_0[0], player_money_round_0[1])
#         counter += 1
#
#     # FLOP, TURN, RIVER
#     player_money_round = player_money_round_0.copy()
#     player_rounds_2_3_4 = []
#     player_rounds_2_3_4.append(player_money_round_0)
#     for action in action_sequence[1:]:
#         player_money_round = player_money_round.copy()
#         counter = 1
#         for a in action:
#             # Player 0 plays first, player 0 plays second
#             # Every odd play is a play by player 0, every even play is a play by player 1
#             if counter % 2 != 0:
#                 # player 0 plays
#                 player_money_round[0] = check_action(a, player_money_round[0], player_money_round[1])
#             else:
#                 # player 1 plays
#                 player_money_round[1] = check_action(a, player_money_round[1], player_money_round[0])
#             counter += 1
#         player_rounds_2_3_4.append(player_money_round)
#
#     new_outer = []
#     for round in player_rounds_2_3_4:
#         new_inner = []
#         for player in round:
#             new_inner.append(player / 20000)
#         new_outer.append(new_inner)
#     return new_outer
#
#
# def check_action(action, player_money, other_player_money):
#     if action == 'a':
#         return 20000
#     if action == 'f':
#         return player_money
#     if action == 'c':
#         return max(player_money, other_player_money)
#     if action == 'p':
#         return max(player_money, other_player_money) + max(player_money, other_player_money) * 2


def gather_data(games):
    dataset = []
    for j in range(len(games)):
        print(games[j])
        for i in range(games[j][0]):
            print(i)
            data, returns = run_game_once(games[j][1], games[j][2])
            if len(data) != 0:
                for turn in data:
                    player = turn[-1]
                    turn[-1] = returns[player]
                    dataset.append(turn)

    return pd.DataFrame(dataset)


def test_agent(enemy, games):
    wins = 0
    losses = 0
    total_money = 0
    for i in range(games):
        data, returns = run_game_once("NN", enemy)
        total_money += returns[0]
        if returns[0] > 0:
            wins += 1
        else:
            losses += 1
    return wins / games, losses / games, total_money, total_money / games


def main(_):

    games = [[500, "random", "NN"],
             [1000, "fold", "NN"],
             [2000, "NN", "NN"]]
    # df = gather_data(games)
    # print(df)
    # df.to_csv("fcpa_dataset_3.csv")

    # run_game_once("fold", "NN")
    #
    win, loss, money, money_per = test_agent("check_call", 250)
    print('win per:', win, ',\nloss per: ', loss, ',\ntotal money: ', money, ',\nmoney per round: ', money_per)


if __name__ == "__main__":
    app.run(main)
