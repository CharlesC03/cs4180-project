from collections import namedtuple
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from env import PokerEnvironment


class ExponentialSchedule:
    def __init__(self, value_from, value_to, num_steps):
        """Exponential schedule from `value_from` to `value_to` in `num_steps` steps.

        $value(t) = a \exp (b t)$

        :param value_from: Initial value
        :param value_to: Final value
        :param num_steps: Number of steps for the exponential schedule
        """
        self.value_from = value_from
        self.value_to = value_to
        self.num_steps = num_steps

        # YOUR CODE HERE: Determine the `a` and `b` parameters such that the schedule is correct
        self.a = self.value_from
        self.b = np.log(self.value_to / self.value_from) / (self.num_steps - 1)

    def value(self, step) -> float:
        """Return exponentially interpolated value between `value_from` and `value_to`interpolated value between.

        Returns {
            `value_from`, if step == 0 or less
            `value_to`, if step == num_steps - 1 or more
            the exponential interpolation between `value_from` and `value_to`, if 0 <= steps < num_steps
        }

        :param step: The step at which to compute the interpolation
        :rtype: Float. The interpolated value
        """

        return self.a * np.exp(self.b * min(max(step, 0), self.num_steps - 1))


# DO NOT EDIT: Test code


def _test_schedule(schedule, step, value, ndigits=5):
    """Tests that the schedule returns the correct value."""
    v = schedule.value(step)
    if not round(v, ndigits) == round(value, ndigits):
        raise Exception(
            f"For step {step}, the scheduler returned {v} instead of {value}"
        )


_schedule = ExponentialSchedule(0.1, 0.2, 3)
_test_schedule(_schedule, -1, 0.1)
_test_schedule(_schedule, 0, 0.1)
_test_schedule(_schedule, 1, 0.141421356237309515)
_test_schedule(_schedule, 2, 0.2)
_test_schedule(_schedule, 3, 0.2)
del _schedule

_schedule = ExponentialSchedule(0.5, 0.1, 5)
_test_schedule(_schedule, -1, 0.5)
_test_schedule(_schedule, 0, 0.5)
_test_schedule(_schedule, 1, 0.33437015248821106)
_test_schedule(_schedule, 2, 0.22360679774997905)
_test_schedule(_schedule, 3, 0.14953487812212207)
_test_schedule(_schedule, 4, 0.1)
_test_schedule(_schedule, 5, 0.1)
del _schedule

# Batch namedtuple, i.e. a class which contains the given attributes
Batch = namedtuple("Batch", ("states", "actions", "rewards", "next_states", "dones"))


class ReplayMemory:
    def __init__(self, max_size: int, state_size: int, num_players: int):
        """Replay memory implemented as a circular buffer.

        Experiences will be removed in a FIFO manner after reaching maximum
        buffer size.

        Args:
            - max_size: Maximum size of the buffer
            - state_size: Size of the state-space features for the environment
        """
        self.max_size = max_size
        self.state_size = state_size
        self.num_players = num_players

        # Preallocating all the required memory, for speed concerns
        # Buffer must be separate for each player
        # self.states = [torch.empty((max_size, state_size)) for _ in range(num_players)]
        self.states = torch.empty((num_players, max_size, state_size))
        self.actions = torch.empty((num_players, max_size, 1), dtype=torch.long)
        self.rewards = torch.empty((num_players, max_size, 1))
        self.next_states = torch.empty((num_players, max_size, state_size))
        self.dones = torch.empty((num_players, max_size, 1), dtype=torch.bool)

        # Pointer to the current location in the circular buffer
        # self.idx = 0
        self.idx = np.zeros(num_players, dtype=int)
        # Indicates number of transitions currently stored in the buffer
        self.size = np.zeros(num_players, dtype=int)

    def add(self, player: int, state, action, reward, next_state, done: bool):
        """Add a transition to the buffer.

        :param state: 1-D np.ndarray of state-features
        :param action: Integer action
        :param reward: Float reward
        :param next_state: 1-D np.ndarray of state-features
        :param done: Boolean value indicating the end of an episode
        """

        # YOUR CODE HERE: Store the input values into the appropriate
        # attributes, using the current buffer position `self.idx`

        self.states[player][self.idx] = torch.tensor(state).float()
        self.actions[player][self.idx] = action
        self.rewards[player][self.idx] = reward
        self.next_states[player][self.idx] = torch.tensor(next_state).float()
        self.dones[player][self.idx] = done

        # DO NOT EDIT
        # Circulate the pointer to the next position
        # self.idx = (self.idx + 1) % self.max_size
        self.idx[player] = (self.idx[player] + 1) % self.max_size
        # Update the current buffer size
        # self.size = min(self.size + 1, self.max_size)
        self.size[player] = min(self.size[player] + 1, self.max_size)

    def sample(self, batch_size) -> Batch:
        # def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of experiences.

        If the buffer contains less that `batch_size` transitions, sample all
        of them.

        :param batch_size: Number of transitions to sample
        :rtype: Batch
        """

        # YOUR CODE HERE: Randomly sample an appropriate number of
        # transitions *without replacement*. If the buffer contains less than
        # `batch_size` transitions, return all of them. The return type must
        # be a `Batch`.

        player = np.random.randint(self.num_players)

        # for player in range(self.num_players):
        #     if self.size < batch_size:
        #         sample_indices = np.arange(self.size[player])
        #     else:
        #         sample_indices = np.random.choice(
        #             self.size[player], batch_size, replace=False
        #         )

        #     batches[f"States (player {player})"] = self.states[player, sample_indices]
        #     batches[f"Actions (player {player})"] = self.actions[player, sample_indices]
        #     batches[f"Rewards (player {player})"] = self.rewards[player, sample_indices]
        #     batches[f"Next States (player {player})"] = self.next_states[
        #         player, sample_indices
        #     ]
        #     batches[f"Dones (player {player})"] = self.dones[player, sample_indices]

        if self.size[player] < batch_size:
            return Batch(
                states=self.states[player][: self.size[player]],
                actions=self.actions[player][: self.size[player]],
                rewards=self.rewards[player][: self.size[player]],
                next_states=self.next_states[player][: self.size[player]],
                dones=self.dones[player][: self.size[player]],
            )

        sample_indices = np.random.choice(self.size[player], batch_size, replace=False)
        batch = Batch(
            states=self.states[player][sample_indices],
            actions=self.actions[player][sample_indices],
            rewards=self.rewards[player][sample_indices],
            next_states=self.next_states[player][sample_indices],
            dones=self.dones[player][sample_indices],
        )

        return batch

    def populate(self, env: PokerEnvironment, num_steps: int):
        """Populate this replay memory with `num_steps` from the random policy.

        :param env: Gymnasium environment
        :param num_steps: Number of steps to populate the replay memory
        """

        # YOUR CODE HERE: Run a random policy for `num_steps` time-steps and
        # populate the replay memory with the resulting transitions.
        # Hint: Use the self.add() method.

        for _ in range(num_steps):
            prev_states = [None] * self.num_players
            player, states = env.reset()
            prev_states[player] = states
            for player in range(self.num_players):
                action = env.random_action()
                player, player_state, rewards, dones = env.step(action)
                prev_states[player] = player_state
            done = False
            while done:
                actions = env.random_action()
                player, player_state, rewards, done = env.step(
                    actions
                )  # MARK: Update this to work with multiple players
                self.add(
                    player,
                    prev_states[player],
                    actions[player],
                    rewards[player],
                    player_state,
                    env.round == 4,  # MARK: This may cause problems
                )
                prev_states[player] = player_state

                states = player_state
        # state = {}
        # player, state, _ = env.reset()
        # for _ in range(num_steps):
        #     action = env.random_action()
        #     player, next_state, reward, done = env.step(action)
        #     self.add(state, action, reward, next_state, done)
        #     if done:
        #         state, _ = env.reset()
        #     else:
        #         state = next_state


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, *, num_layers=20, hidden_dim=256):
        """Deep Q-Network PyTorch model.

        Args:
            - state_dim: Dimensionality of states
            - action_dim: Dimensionality of actions
            - num_layers: Number of total linear layers
            - hidden_dim: Number of neurons in the hidden layers
        """

        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # YOUR CODE HERE: Define the layers of your model such that
        # * there are `num_layers` nn.Linear modules / layers
        # * all activations except the last should be ReLU activations
        #   (this can be achieved either using a nn.ReLU() object or the nn.functional.relu() method)
        # * the last activation can either be missing, or you can use nn.Identity()
        # Hint: A regular Python list of layers is tempting, but PyTorch does not register
        # these parameters in its computation graph. See nn.ModuleList or nn.Sequential
        self.in_layer = nn.Linear(self.state_dim, self.hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()]
            * (self.num_layers - 1)
        )
        self.final_hidden_layer = nn.Linear(self.hidden_dim, self.action_dim)

    def forward(self, states) -> torch.Tensor:
        """Q function mapping from states to action-values.

        :param states: (*, S) torch.Tensor where * is any number of additional
                dimensions, and S is the dimensionality of state-space
        :rtype: (*, A) torch.Tensor where * is the same number of additional
                dimensions as the `states`, and A is the dimensionality of the
                action-space. This represents the Q values Q(s, .)
        """
        # YOUR CODE HERE: Use the defined layers and activations to compute
        # the action-values tensor associated with the input states.
        # Hint: Do not worry about the * arguments above (previous dims in tensor).
        # PyTorch functions typically handle those properly.

        # batch_size = states.size(0)
        # states = states.view(batch_size * self.num_players, self.state_dim)

        # Process the state-action mapping individually for each player
        # outputs = []
        # for player in range(self.num_players):
        #     x = states[player :: self.num_players]
        #     x = self.in_layers[player](x)
        #     x = F.relu(x)
        #     for layer in self.hidden_layers[
        #         player * (self.num_layers - 1) : (player + 1) * (self.num_layers - 1)
        #     ]:
        #         x = layer[0](x)
        #         x = layer[1](x)
        #     x = self.final_hidden_layer[player](x)
        #     outputs.append(x)

        # # stacks and reshapes the tensor to fit the desired shape
        # return torch.stack(outputs, dim=1).view(
        #     batch_size, self.num_players, self.action_dim
        # )

        x = self.in_layer(states)
        x = F.relu(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.final_hidden_layer(x)
        x = F.softmax(x, dim=1)
        x = nn.Identity()(x)
        return x

    # DO NOT EDIT: Utility methods for cloning and storing models.

    @classmethod
    def custom_load(cls, data):
        model = cls(*data["args"], **data["kwargs"])
        model.load_state_dict(data["state_dict"])
        return model

    def custom_dump(self):
        return {
            "args": (self.state_dim, self.action_dim),
            "kwargs": {
                "num_layers": self.num_layers,
                "hidden_dim": self.hidden_dim,
            },
            "state_dict": self.state_dict(),
        }


# DO NOT EDIT: Test code


def _test_dqn_forward(dqn_model, input_shape, output_shape):
    """Tests that the dqn returns the correctly shaped tensors."""
    inputs = torch.torch.randn((input_shape))
    outputs = dqn_model(inputs)

    if not isinstance(outputs, torch.FloatTensor):
        raise Exception(
            f"DQN.forward returned type {type(outputs)} instead of torch.Tensor"
        )

    if outputs.shape != output_shape:
        raise Exception(
            f"DQN.forward returned tensor with shape {outputs.shape} instead of {output_shape}"
        )

    if not outputs.requires_grad:
        raise Exception(
            "DQN.forward returned tensor which does not require a gradient (but it should)"
        )


dqn_model = DQN(10, 4)
_test_dqn_forward(dqn_model, (64, 10), (64, 4))
_test_dqn_forward(dqn_model, (2, 3, 10), (2, 3, 4))
del dqn_model

dqn_model = DQN(64, 16)
_test_dqn_forward(dqn_model, (64, 64), (64, 16))
_test_dqn_forward(dqn_model, (2, 3, 64), (2, 3, 16))
del dqn_model

# Testing custom dump / load
dqn1 = DQN(10, 4, num_layers=10, hidden_dim=20)
dqn2 = DQN.custom_load(dqn1.custom_dump())
assert dqn2.state_dim == 10
assert dqn2.action_dim == 4
assert dqn2.num_layers == 10
assert dqn2.hidden_dim == 20


def train_dqn_batch(optimizer, batch, dqn_model, dqn_target, gamma) -> float:
    """Perform a single batch-update step on the given DQN model.

    :param optimizer: nn.optim.Optimizer instance
    :param batch: Batch of experiences (class defined earlier)
    :param dqn_model: The DQN model to be trained
    :param dqn_target: The target DQN model, ~NOT~ to be trained
    :param gamma: The discount factor
    :rtype: Float. The scalar loss associated with this batch
    """
    # YOUR CODE HERE: Compute the values and target_values tensors using the
    # given models and the batch of data.
    # Recall that 'Batch' is a named tuple consisting of
    # ('states', 'actions', 'rewards', 'next_states', 'dones')
    # Hint: Remember that we should not pass gradients through the target network
    # dqn_model.train()
    # MARK: TODO: Upate this part to work with multiple players
    values = dqn_model(batch.states).gather(1, batch.actions)
    target_values = batch.rewards + gamma * torch.max(
        dqn_target(batch.next_states).detach(), dim=1
    ).values.unsqueeze(1) * (1 - batch.dones.int())

    # DO NOT EDIT

    assert (
        values.shape == target_values.shape
    ), "Shapes of values tensor and target_values tensor do not match."

    # Testing that the values tensor requires a gradient,
    # and the target_values tensor does not
    assert values.requires_grad, "values tensor requires gradients"
    assert (
        not target_values.requires_grad
    ), "target_values tensor should not require gradients"

    # Computing the scalar MSE loss between computed values and the TD-target
    # DQN originally used Huber loss, which is less sensitive to outliers
    loss = F.mse_loss(values, target_values)

    optimizer.zero_grad()  # Reset all previous gradients
    loss.backward()  # Compute new gradients
    optimizer.step()  # Perform one gradient-descent step

    return loss.item()


def train_dqn(
    env: PokerEnvironment,
    num_steps,
    model,
    *,
    model_num_layers=20,
    model_hidden_dim=256,
    num_saves=5,
    replay_size,
    replay_prepopulate_steps=0,
    batch_size,
    exploration,
    gamma,
):
    """
    DQN algorithm.

    Compared to previous training procedures, we will train for a given number
    of time-steps rather than a given number of episodes. The number of
    time-steps will be in the range of millions, which still results in many
    episodes being executed.

    Args:
        - env: The Gymnasium environment
        - num_steps: Total number of steps to be used for training
        - num_saves: How many models to save to analyze the training progress
        - replay_size: Maximum size of the ReplayMemory
        - replay_prepopulate_steps: Number of steps with which to prepopulate
                                    the memory
        - batch_size: Number of experiences in a batch
        - exploration: An ExponentialSchedule
        - gamma: The discount factor

    Returns: (saved_models, returns)
        - saved_models: Dictionary whose values are trained DQN models
        - returns: Numpy array containing the return of each training episode
        - lengths: Numpy array containing the length of each training episode
        - losses: Numpy array containing the loss of each training batch
    """
    # Check that environment states are compatible with our DQN representation
    # assert (
    #     isinstance(env.observation_space, gym.spaces.Box)
    #     and len(env.observation_space.shape) == 1
    # )

    # assert len(env.state_shape) == 1

    # Get the state_size from the environment
    state_size = env.state_size

    # Initialize the DQN and DQN-target models
    dqn_model = model(
        state_size,
        env.action_space,
        num_layers=model_num_layers,
        hidden_dim=model_hidden_dim,
    )
    dqn_models = [copy.deepcopy(dqn_model) for _ in range(env.num_players)]

    dqn_target = model.custom_load(dqn_model.custom_dump())

    # Initialize the optimizer
    optimizer = torch.optim.Adam(dqn_model.parameters())

    # Initialize the replay memory and prepopulate it
    memory = ReplayMemory(replay_size, state_size, 2)
    memory.populate(env, replay_prepopulate_steps)

    # Initialize lists to store returns, lengths, and losses
    # rewards = [[] for _ in range(env.num_players)]
    returns = [[] for _ in range(env.num_players)]
    lengths = []
    losses = []

    # Initialize structures to store the models at different stages of training
    t_saves = np.linspace(0, num_steps, num_saves - 1, endpoint=False)
    saved_models = {}

    i_episode = 0  # Use this to indicate the index of the current episode
    t_episode = 0  # Use this to indicate the time-step inside current episode

    prev_states = np.zeros((env.num_players, state_size))

    player, state = env.reset()  # Initialize state of first episode

    prev_states[player] = state
    G = np.zeros(env.num_players, dtype=float)

    # Iterate for a total of `num_steps` steps
    pbar = tqdm.trange(num_steps)
    for t_total in pbar:
        # Use t_total to indicate the time-step from the beginning of training

        # Save model
        if t_total in t_saves:
            model_name = f"{100 * t_total / num_steps:04.1f}".replace(".", "_")
            saved_models[model_name] = copy.deepcopy(dqn_model)

        # YOUR CODE HERE:
        #  * sample an action from the DQN using epsilon-greedy
        #  * use the action to advance the environment by one step
        #  * store the transition into the replay memory
        action = (
            env.random_action()
            if np.random.rand() < exploration.value(t_total)
            else torch.argmax(
                dqn_model(torch.tensor(state).float().unsqueeze(0))
            ).item()
        )

        player, next_state, reward, done = env.step(action)

        memory.add(
            player, prev_states[player], action, reward, next_state, env.round == 4
        )
        G[player] = reward + gamma * G[player]
        # MARK: Add support to update individually for each of the players
        # rewards[player].append(reward)
        # YOUR CODE HERE: Once every 4 steps,
        #  * sample a batch from the replay memory
        #  * perform a batch update (use the train_dqn_batch() method)

        if t_total % 4 == 0:
            batch = memory.sample(batch_size)
            loss = train_dqn_batch(optimizer, batch, dqn_model, dqn_target, gamma)
            losses.append(loss)

        # YOUR CODE HERE: Once every 10_000 steps,
        #  * update the target network (use the dqn_model.state_dict() and
        #    dqn_target.load_state_dict() methods)

        if t_total % 10_000 == 0:
            dqn_target.load_state_dict(dqn_model.state_dict())

        if done:
            # YOUR CODE HERE: Anything you need to do at the end of an episode,
            # e.g., compute return G, store returns/lengths,
            # reset variables, indices, lists, etc.
            # G = sum([gamma**i * rewards[i] for i in range(len(rewards))])
            for i in range(env.num_players):
                returns[i].append(G[i])
            eps = exploration.value(t_total)
            lengths.append(t_episode)

            pbar.set_description(
                f"Episode: {i_episode} | Steps: {t_episode + 1} | Return: {G.mean():5.2f}({G.min():5.2f} {G.max():5.2f}) | Epsilon: {eps:4.2f}"
            )

            if env.game_over:
                env.full_reset()

            player, state = env.reset()
            prev_states[player] = state
            i_episode += 1
            t_episode = 0
            # rewards = [[] for _ in range(env.num_players)]
            G = np.zeros(env.num_players, dtype=float)
        else:
            prev_states[player] = next_state
            t_episode += 1

    saved_models["100_0"] = copy.deepcopy(dqn_model)

    return (
        saved_models,
        np.array(returns),
        np.array(lengths),
        np.array(losses),
    )
