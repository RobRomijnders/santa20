import random
from collections import Counter
import numpy as np

actions = []
rewards = []
opponent_picks = []
counts = {}
prior = {"n": 1, "h": 1}


def oppo_action(last_actions, my_action):
    a1, a2 = last_actions 
    if a1 == my_action:
        return int(a2)
    return int(a1)
    

def multi_armed_bandit_agent(observation, configuration):
    global actions, rewards, counts
    if len(counts) == 0:
        for i in range(configuration.banditCount):
            counts[i] = prior
    if len(observation.lastActions) > 0:
        rewards.append(observation.reward)
        opponent_picks.append(oppo_action(observation.lastActions, actions[-1]))
        reward_t2 = rewards[-2] if len(rewards) >= 2 else 0
        reward_t1 = rewards[-1] if len(rewards) > 0 else 0
        counts[actions[-1]] = {
            "n": counts[actions[-1]]["n"] + 1, 
            "h": counts[actions[-1]]["h"] + (reward_t1 - reward_t2)
        }
    
    action = random.randrange(configuration.banditCount)
    if observation.step > 1:
        action = oppo_action(observation.lastActions, actions[-1])

    if observation.step > 10:
        pvals = np.array([np.random.beta(d['n'], d['h']) for d in counts.values()])
        pvals = pvals/pvals.sum()
        action = int(np.random.choice([i for i in range(len(counts))], p=pvals/pvals.sum()))

    actions.append(action)
    return action