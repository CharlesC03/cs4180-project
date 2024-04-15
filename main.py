from env import PokerEnvironment
from agent import DQNAgent

if __name__ == "__main__":
    env = PokerEnvironment()
    state_shape = (len(env.reset()),)
    action_size = 4
    
    agent = DQNAgent(state_shape, action_size)
    agent.train(env, episodes=10)

