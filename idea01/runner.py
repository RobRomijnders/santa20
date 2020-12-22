from bots import Bots

from kaggle_environments import make
import tqdm


def best_of_n(agent1, agent2, n=20):
    env = make("mab", debug=True)
    wins = []
    for i in tqdm.tqdm(range(n)):
        env.run([agent1, agent2])
        p1_score = env.steps[-1][0]['reward']
        p2_score = env.steps[-1][1]['reward']
        env.reset()
        wins.append(p1_score > p2_score)
        print(f"Round {i+1}: {p1_score} - {p2_score}")
    print(f"{round(sum(wins)/len(wins), ndigits=4)}")


if __name__ == "__main__":
    best_of_n(Bots().random_agent, Bots().thompson_sampling_agent)
