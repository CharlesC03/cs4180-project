import torch

from agent import DQN
from env import PokerEnvironment

action_str_mapping = {0: "Fold", 1: "Call", 2: "Raise", 3: "All in"}


def render(env: PokerEnvironment, policy=None, pvp=False, mutiple_rounds=False):
    """Graphically render an episode using the given policy

    :param env: Gymnasium environment
    :param policy: Function which maps state to action.  If None, the random
                   policy is used.
    """

    if policy is None:
        # Random policy
        def policy(state):
            return env.random_action()

    # Basic gym loop
    player, state = env.reset()
    if not pvp or player == 0:
        env.render()
    while True:
        if pvp and player == 0 and env.round != 4:
            action = int(input("Enter action(Fold: 0, Call: 1, Raise: 2, All in: 3): "))
        else:
            action = policy(state)
            print(f"AI action: {action_str_mapping[action]}")
        player, next_state, reward, terminated = env.step(action)
        # if not pvp or env.round == 4 or player == 0:
        print(
            f"Player: {player}, State: {next_state}, Reward: {reward}, Terminated: {terminated}"
        )
        env.render()
        state = next_state
        if terminated:
            if not mutiple_rounds or env.game_over:
                break
            else:
                player, state = env.reset()
                # if not pvp or env.round == 4 or player == 0:
                env.render()


buttons_all = []
model_class = DQN(18, 4)
env = PokerEnvironment()
# PVP = False
MODEL = "./models/checkpoint_poker_bug_fix_8x512_1500000.pt"
PERCENTAGE = "100_0"
try:
    checkpoint = torch.load(MODEL)
except FileNotFoundError:
    print("No checkpoint found")
    pass
else:
    dqn = model_class.custom_load(checkpoint[PERCENTAGE])
    render(
        env,
        lambda state: dqn(torch.tensor(state, dtype=torch.float).unsqueeze(0))
        .argmax()
        .item(),
        pvp=True,
        mutiple_rounds=True,
    )
