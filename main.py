import numpy as np
from game import Board
from game import IllegalAction, GameOver
from agent import nTupleNewrok
import pickle

from collections import namedtuple

"""
Vocabulary
--------------

Transition: A Transition shows how a board transfromed from a state to the next state. It contains the board state (s), the action performed (a), 
the reward received by performing the action (r), the board's "after state" after applying the action (s_after), and the board's "next state" (s_next) after adding a random tile to the "after state".

Gameplay: A series of transitions on the board (transition_history). Also reports the total reward of playing the game (game_reward) and the maximum tile reached (max_tile).
"""
Transition = namedtuple("Transition", "s, a, r, s_after, s_next")
Gameplay = namedtuple("Gameplay", "transition_history game_reward max_tile")


def play(agent, board, spawn_random_tile=False):
    "Return a gameplay of playing the given (board) until terminal states."
    b = Board(board)
    r_game = 0
    a_cnt = 0
    transition_history = []
    while True:
        a_best = agent.best_action(b.board)
        s = b.copyboard()
        try:
            r = b.act(a_best)
            r_game += r
            s_after = b.copyboard()
            b.spawn_tile(random_tile=spawn_random_tile)
            s_next = b.copyboard()
        except (IllegalAction, GameOver) as e:
            # game ends when agent makes illegal moves or board is full
            r = None
            s_after = None
            s_next = None
            break
        finally:
            a_cnt += 1
            transition_history.append(
                Transition(s=s, a=a_best, r=r, s_after=s_after, s_next=s_next)
            )
    gp = Gameplay(
        transition_history=transition_history,
        game_reward=r_game,
        max_tile=2 ** max(b.board),
    )
    learn_from_gameplay(agent, gp)
    return gp


def learn_from_gameplay(agent, gp, alpha=0.1):
    "Learn transitions in reverse order except the terminal transition"
    for tr in gp.transition_history[:-1][::-1]:
        agent.learn(tr.s, tr.a, tr.r, tr.s_after, tr.s_next, alpha=alpha)


def load_agent(path):
    return pickle.load(path.open("rb"))


# map board state to LUT
TUPLES = [
    # horizontal 4-tuples
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
    [12, 13, 14, 15],
    # vertical 4-tuples
    [0, 4, 8, 12],
    [1, 5, 9, 13],
    [2, 6, 10, 14],
    [3, 7, 11, 15],
    # all 4-tile squares
    [0, 1, 4, 5],
    [4, 5, 8, 9],
    [8, 9, 12, 13],
    [1, 2, 5, 6],
    [5, 6, 9, 10],
    [9, 10, 13, 14],
    [2, 3, 6, 7],
    [6, 7, 10, 11],
    [10, 11, 14, 15],
]

if __name__ == "__main__":
    import numpy as np

    agent = None
    # prompt to load saved agents
    from pathlib import Path

    path = Path("tmp")
    saves = list(path.glob("*.pkl"))
    if len(saves) > 0:
        print("Found %d saved agents:" % len(saves))
        for i, f in enumerate(saves):
            print("{:2d} - {}".format(i, str(f)))
        k = input(
            "input the id to load an agent, input nothing to create a fresh agent:"
        )
        if k.strip() != "":
            k = int(k)
            n_games, agent = load_agent(saves[k])
            print("load agent {}, {} games played".format(saves[k].stem, n_games))
    if agent is None:
        print("initialize agent")
        n_games = 0
        agent = nTupleNewrok(TUPLES)

    n_session = 5000
    n_episode = 100
    print("training")
    try:
        for i_se in range(n_session):
            gameplays = []
            for i_ep in range(n_episode):
                gp = play(agent, None, spawn_random_tile=True)
                gameplays.append(gp)
                n_games += 1
            n2048 = sum([1 for gp in gameplays if gp.max_tile == 2048])
            mean_maxtile = np.mean([gp.max_tile for gp in gameplays])
            maxtile = max([gp.max_tile for gp in gameplays])
            mean_gamerewards = np.mean([gp.game_reward for gp in gameplays])
            print(
                "Game# %d, tot. %dk games, " % (n_games, n_games / 1000)
                + "mean game rewards {:.0f}, mean max tile {:.0f}, 2048 rate {:.0%}, maxtile {}".format(
                    mean_gamerewards, mean_maxtile, n2048 / len(gameplays), maxtile
                ),
            )
    except KeyboardInterrupt:
        print("training interrupted")
        print("{} games played by the agent".format(n_games))
        if input("save the agent? (y/n)") == "y":
            fout = "tmp/{}_{}games.pkl".format(agent.__class__.__name__, n_games)
            pickle.dump((n_games, agent), open(fout, "wb"))
            print("agent saved to", fout)
