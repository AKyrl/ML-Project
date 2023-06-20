#!/usr/bin/env python3
# encoding: utf-8
"""
fcpa_agent.py

Extend this class to provide an agent that can participate in a tournament.

Created by Pieter Robberechts, Wannes Meert.
Copyright (c) 2021 KU Leuven. All rights reserved.
"""
import os
import sys
import argparse
import logging
import numpy as np
import tensorflow as tf
import pyspiel
from keras.models import load_model
from open_spiel.python.algorithms import evaluate_bots

from fcpa_train import chen, evaluate

logger = logging.getLogger('be.kuleuven.cs.dtai.fcpa')
global model


def get_agent_for_tournament(player_id):
    """Change this function to initialize your agent.
    This function is called by the tournament code at the beginning of the
    tournament.

    :param player_id: The integer id of the player for this bot, e.g. `0` if
        acting as the first player.
    """
    my_player = Agent(player_id)
    return my_player


def from_state_to_inputs(state):
    cards, board_cards, rounds, spent0, spent1, prev_action = parse_state_data(state)

    chen0 = chen(cards)

    eval0 = evaluate(cards + board_cards)

    return [chen0, eval0, spent0, spent1, prev_action, rounds]


def remove_useless_data(state):
    state = str(state).split('\n')
    state.pop(0)
    state.pop(4)
    state.pop(5)
    return state


def parse_state_data(state):
    state = remove_useless_data(state)

    p0cards = state[0].split(':')[1][1:]
    p0cards = [p0cards[:2], p0cards[2:]]

    p1cards = state[1].split(':')[1][1:]
    p1cards = [p1cards[:2], p1cards[2:]]

    board_cards = state[2].split(' ')[1]
    board_cards = fromStringToCardList(board_cards)

    player = int(state[3][-1:])

    rounds = int(state[4].split(' ')[1])

    spent = state[5].split('[')[1]
    spent = spent.split(' ')
    spent0 = spent[1]
    spent1 = spent[4]

    actions = state[7][-1:]

    if actions == 'f':
        prev_action = 0
    elif actions == 'c':
        prev_action = 1
    elif actions == 'p':
        prev_action = 2
    elif actions == 'a':
        prev_action = 3
    else:
        prev_action = 4

    if player == 1:
        cards = p1cards
    else:
        cards = p0cards

    return cards, board_cards, rounds, int(spent0) / 20000, int(spent1) / 20000, prev_action



def parse_acpc(acpc_state):
    cards = acpc_state.split(':')[1]
    cards = cards.split('/')
    cards.pop(0)
    return cards


def get_last_action(curr_round, action_sequence):
    curr_round_action_seq = action_sequence[curr_round]
    twoLast = ''
    if len(curr_round_action_seq) >= 2:
        twoLast = curr_round_action_seq[-2:]
    else:
        return twoLast, twoLast
    if curr_round == 0:
        if len(curr_round_action_seq) % 2 == 0:
            return twoLast[1], twoLast[0]
        else:
            return twoLast[0], twoLast[1]
    else:
        if len(curr_round_action_seq) % 2 == 0:
            return twoLast[0], twoLast[1]
        else:
            return twoLast[1], twoLast[0]


def fromStringToCardList(board_cards):
    board = []
    if len(board_cards) == 0:
        pass
    elif len(board_cards) == 6:
        board.append(board_cards[:2])
        board.append(board_cards[2:4])
        board.append(board_cards[4:6])
    elif len(board_cards) == 8:
        board.append(board_cards[:2])
        board.append(board_cards[2:4])
        board.append(board_cards[4:6])
        board.append(board_cards[6:8])
    elif len(board_cards) == 10:
        board.append(board_cards[:2])
        board.append(board_cards[2:4])
        board.append(board_cards[4:6])
        board.append(board_cards[6:8])
        board.append(board_cards[8:10])
    return board


class Agent(pyspiel.Bot):
    """Agent template"""

    def __init__(self, player_id):
        """Initialize an agent to play FCPA poker.

        Note: This agent should make use of a pre-trained policy to enter
        the tournament. Initializing the agent should thus take no more than
        a few seconds.
        """
        global model
        package_directory = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(package_directory, 'fcpa_nn_dummy')
        model = load_model(model_path)
        pyspiel.Bot.__init__(self)
        self.player_id = player_id

    def restart_at(self, state):
        """Starting a new game in the given state.

        :param state: The initial state of the game.
        """
        pass

    def inform_action(self, state, player_id, action):
        """Let the bot know of the other agent's actions.

        :param state: The current state of the game.
        :param player_id: The ID of the player that executed an action.
        :param action: The action which the player executed.
        """
        pass

    def step(self, state):
        """Returns the selected action in the given state.

        :param state: The current state of the game.
        :returns: The selected action from the legal actions, or
            `pyspiel.INVALID_ACTION` if there are no legal actions available.
        """
        # print(state.legal_actions())
        data = from_state_to_inputs(state)

        data = np.asarray(data).astype('float32')
        # print(data)

        y = model.predict(tf.expand_dims(data, axis=0))

        # print(y[0])
        out = y[0]
        move = 0
        val = 0
        for i in range(len(out)):
            if out[i] >= val and i in state.legal_actions():
                val = out[i]
                move = i

        # print(move)
        return move


def test_api_calls():
    """This method calls a number of API calls that are required for the
    tournament. It should not trigger any Exceptions.
    """
    fcpa_game_string = (
        "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=150 100,"
        "firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,"
        "stack=20000 20000,bettingAbstraction=fcpa)")
    game = pyspiel.load_game(fcpa_game_string)
    bots = [get_agent_for_tournament(player_id) for player_id in [0, 1]]
    returns = evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)
    assert len(returns) == 2
    assert isinstance(returns[0], float)
    assert isinstance(returns[1], float)
    print("SUCCESS!")


def main(argv=None):
    test_api_calls()


if __name__ == "__main__":
    sys.exit(main())
