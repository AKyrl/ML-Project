import pyspiel

def subsidy():
    """Creates subsidy manually from the spiel building blocks."""

    game_type = pyspiel.GameType(
        "subsidy",
        "Subsidy",
        pyspiel.GameType.Dynamics.SIMULTANEOUS,
        pyspiel.GameType.ChanceMode.DETERMINISTIC,
        pyspiel.GameType.Information.ONE_SHOT,
        pyspiel.GameType.Utility.GENERAL_SUM,
        pyspiel.GameType.RewardModel.TERMINAL,
        2,  # max num players
        2,  # min_num_players
        True,  # provides_information_state
        True,  # provides_information_state_tensor
        False,  # provides_observation
        False,  # provides_observation_tensor
        dict()  # parameter_specification
    )
    game = pyspiel.MatrixGame(
        game_type,
        {},  # game_parameters
        ["S1", "S2"],  # row_action_names
        ["S1", "S2"],  # col_action_names
        [[10, 0], [11, 12]],  # row player utilities
        [[10, 11], [0, 12]]  # col player utilities
    )
    return game

def battle_of_the_sexes():
    """Creates battle of the sexes manually from the spiel building blocks."""

    game_type = pyspiel.GameType(
        "battle_of_the_sexes",
        "Battle of the Sexes",
        pyspiel.GameType.Dynamics.SIMULTANEOUS,
        pyspiel.GameType.ChanceMode.DETERMINISTIC,
        pyspiel.GameType.Information.ONE_SHOT,
        pyspiel.GameType.Utility.GENERAL_SUM,
        pyspiel.GameType.RewardModel.TERMINAL,
        2,  # max num players
        2,  # min_num_players
        True,  # provides_information_state
        True,  # provides_information_state_tensor
        False,  # provides_observation
        False,  # provides_observation_tensor
        dict()  # parameter_specification
    )
    game = pyspiel.MatrixGame(
        game_type,
        {},  # game_parameters
        ["Boxing", "Ballet"],  # row_action_names
        ["Boxing", "Ballet"],  # col_action_names
        [[3, 0], [0, 2]],  # row player utilities
        [[2, 0], [0, 3]]  # col player utilities
    )
    return game

def dispersion():
    """Creates dispersion manually from the spiel building blocks."""

    game_type = pyspiel.GameType(
        "dispersion",
        "Dispersion",
        pyspiel.GameType.Dynamics.SIMULTANEOUS,
        pyspiel.GameType.ChanceMode.DETERMINISTIC,
        pyspiel.GameType.Information.ONE_SHOT,
        pyspiel.GameType.Utility.GENERAL_SUM,
        pyspiel.GameType.RewardModel.TERMINAL,
        2,  # max num players
        2,  # min_num_players
        True,  # provides_information_state
        True,  # provides_information_state_tensor
        False,  # provides_observation
        False,  # provides_observation_tensor
        dict()  # parameter_specification
    )
    game = pyspiel.MatrixGame(
        game_type,
        {},  # game_parameters
        ["Action A", "Action B"],  # row_action_names
        ["Action A", "Action B"],  # col_action_names
        [[-1, 1], [1, -1]],  # row player utilities
        [[-1, 1], [1, -1]]  # col player utilities
    )
    return game


def biased_rps():
    """Creates biased rock paper scissors manually from the spiel building blocks."""

    game_type = pyspiel.GameType(
        "biased_rps",
        "Biased Rock Paper Scissors",
        pyspiel.GameType.Dynamics.SIMULTANEOUS,
        pyspiel.GameType.ChanceMode.DETERMINISTIC,
        pyspiel.GameType.Information.ONE_SHOT,
        pyspiel.GameType.Utility.ZERO_SUM,
        pyspiel.GameType.RewardModel.TERMINAL,
        2,  # max num players
        2,  # min_num_players
        True,  # provides_information_state
        True,  # provides_information_state_tensor
        False,  # provides_observation
        False,  # provides_observation_tensor
        dict()  # parameter_specification
    )
    game = pyspiel.MatrixGame(
        game_type,
        {},  # game_parameters
        ["Rock", "Paper", "Scissors"],  # row_action_names
        ["Rock", "Paper", "Scissors"],  # col_action_names
        [[0, -0.25, 0.5], [0.25, 0, -0.05], [-0.5, 0.05, 0]],  # row player utilities
        [[0, 0.25, -0.5], [-0.25, 0, 0.05], [0.5, -0.05, 0]]  # col player utilities
    )
    return game
