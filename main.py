from env import PokerEnvironment
from agent import DQN, ExponentialSchedule, train_dqn
import torch

if __name__ == "__main__":
    # train_dqn
    # env = PokerEnvironment()
    # state_shape = (len(env.reset()),)
    # action_size = 4

    # agent = DQNAgent(state_shape, action_size)
    # agent.train(env, episodes=10)

    env = PokerEnvironment()
    gamma = 0.99

    # We train for many time-steps; as usual, you can decrease this during development / debugging,
    # but make sure to restore it to 1_500_000 before submitting
    num_steps = 100_000
    num_saves = 5  # Save models at 0%, 25%, 50%, 75% and 100% of training

    replay_size = 200_000
    replay_prepopulate_steps = 50_000

    batch_size = 64
    exploration = ExponentialSchedule(1.0, 0.05, 1_000_000)

    model_class = DQN

    # This should take about 1-2 hours on a generic 4-core laptop
    dqn_models, returns, lengths, losses = train_dqn(
        env,
        num_steps,
        num_saves=num_saves,
        model=model_class,
        replay_size=replay_size,
        replay_prepopulate_steps=replay_prepopulate_steps,
        batch_size=batch_size,
        exploration=exploration,
        gamma=gamma,
    )

    assert len(dqn_models) == num_saves
    assert all(isinstance(value, model_class) for value in dqn_models.values())

    # Saving computed models to disk, so that we can load and visualize them later
    checkpoint = {key: dqn.custom_dump() for key, dqn in dqn_models.items()}
    torch.save(checkpoint, f"checkpoint_{env.spec.id}.pt")
