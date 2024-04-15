import random
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np

class DQNAgent:
    def __init__(self, state_shape, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_shape = state_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma 
        self.epsilon = epsilon 
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = []
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    # Define the neural network model
    def build_model(self):
        model = models.Sequential([
            layers.Dense(24, input_shape=self.state_shape, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model

    # Update the target model with the weights of the main model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Store the state, action, reward, next_state, and done in the memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Choose an action based on the epsilon-greedy policy
    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.predict(np.expand_dims(state, axis=0))
        return np.argmax(q_values[0])

    # Train the model using the experience replay technique
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            try:
                # Preprocess state
                processed_state = np.array(state)
                processed_next_state = np.array(next_state)
            except ValueError:
                processed_state = [np.array(sub_state) for sub_state in state]
                processed_next_state = [np.array(sub_next_state) for sub_next_state in next_state]

            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(processed_next_state)[0])

            target_f = self.model.predict(processed_state)
            target_f[0][action] = target
            self.model.fit(processed_state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # Train the model for a given number of episodes
    def train(self, env, episodes, batch_size=32):
        for episode in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                self.replay(batch_size)
            self.update_target_model()
            print(f"Episode: {episode + 1}, Epsilon: {self.epsilon}")
