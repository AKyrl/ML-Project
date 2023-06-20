from open_spiel.python.egt import dynamics
from open_spiel.python.egt import utils
import open_spiel.python.egt.visualization as visualization
import matplotlib.pyplot as plt
from matplotlib import projections


def plot_replicator_dynamics_3x3(game, history):
    projections.register_projection(visualization.Dynamics3x3Axes)
    payoff_tensor = utils.game_payoffs_array(game)
    dyn = dynamics.SinglePopulationDynamics(payoff_tensor, dynamics.replicator)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3x3")
    ax.quiver(dyn)
    ax.set_labels(["Rock", "Paper", "Scissors"])
    ax.set_title(f"{game.get_type().long_name} trajectory plot")
    plt.savefig("img/trajectory_plot_" + game.get_type().short_name + ".png")
    plt.show()

    # fig = plt.figure(figsize=(4, 4))
    # ax = fig.add_subplot(111, projection="3x3")
    # ax.streamplot(dyn)
    # ax.set_labels(["Rock", "Paper", "Scissors"])
    # ax.set_title(f"{game.get_type().long_name} directional field plot")
    # plt.savefig("img/streamline_" + game.get_type().short_name + ".png")
    # plt.show()

    return

def plot_replicator_dynamics_2x2(game, history):
    projections.register_projection(visualization.Dynamics2x2Axes)
    payoff_tensor = utils.game_payoffs_array(game)
    dyn = dynamics.MultiPopulationDynamics(payoff_tensor, dynamics.replicator)
    #
    # fig = plt.figure(figsize=(4, 4))
    # ax = fig.add_subplot(111, projection='2x2')
    # ax.quiver(dyn)
    # for pop_hist in history:
    #     x = [hist[0][0] for hist in pop_hist]  # take the prob of choosing the first action for player 1
    #     y = [hist[1][0] for hist in pop_hist]  # take the prob of choosing the first action for player 2
    #     plt.plot(x, y)  # plot each population
    # ax.set_title(f"{game.get_type().long_name} trajectory plot")
    # ax.set_xlabel("Player 1: Prob of choosing A")
    # ax.set_ylabel("Player 2: Prob of choosing A")

    # plt.savefig("img/trajectory_plot_" + game.get_type().short_name + ".png")
    # plt.show()
    #
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='2x2')
    ax.streamplot(dyn)
    ax.set_title(f"{game.get_type().long_name} directional field plot")
    ax.set_xlabel("Player 1: Prob of choosing A")
    ax.set_ylabel("Player 2: Prob of choosing A")

    plt.savefig("img/streamline_" + game.get_type().short_name + ".png")
    # plt.show()
    return
