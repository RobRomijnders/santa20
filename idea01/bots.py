import random
import numpy as np


def oppo_action(last_actions, my_action):
    action1, action2 = last_actions
    if action1 == my_action:
        return int(action2)
    return int(action1)


class Bots:
    def __init__(self, prior=(10, 10), decay=0.9, limit=1.0):
        self.prior = {"n": prior[0], "h": prior[1]}
        self.decay = decay
        self.limit = limit
        self.actions = []
        self.rewards = []
        self.opponent_picks = []
        self.counts = {}

    @staticmethod
    def random_agent(observation, configuration):
        """Agent for taking a random action."""
        del observation
        return random.randrange(configuration.banditCount)

    def random_agent_limit(self, observation, configuration):
        """Agent for taking a random action within a limit."""
        del observation
        return random.randrange(int(configuration.banditCount * self.limit))

    def random_agent_constant(self, observation, configuration):
        """Just returns the same value over and over again."""
        del observation
        return int(configuration.banditCount * self.limit)

    def thompson_sampling_agent(self, observation, configuration):
        """Agent that uses Thompson sampling."""
        if len(self.counts) == 0:
            # Initialize the prior distribution
            for i in range(configuration.banditCount):
                self.counts[i] = self.prior

        if len(observation.lastActions) > 0:
            self.rewards.append(observation.reward)
            self.opponent_picks.append(
                oppo_action(observation.lastActions, self.actions[-1]))

            reward_t2 = self.rewards[-2] if len(self.rewards) >= 2 else 0
            reward_t1 = self.rewards[-1] if len(self.rewards) > 0 else 0

            self.counts[self.actions[-1]] = {
                "n": self.counts[self.actions[-1]]["n"] + 1,
                "h": (self.counts[self.actions[-1]]["h"] +
                      (reward_t1 - reward_t2))
            }

        action = random.randrange(configuration.banditCount)
        if observation.step > 1:
            action = oppo_action(observation.lastActions, self.actions[-1])

        if observation.step > 10:
            pvals = np.array([np.random.beta(d['n'], max(0, d['h']))
                              for d in self.counts.values()])
            pvals = pvals / pvals.sum()
            action = int(np.random.choice(list(range(len(self.counts))),
                                          p=pvals / pvals.sum()))

        self.actions.append(action)
        return action
