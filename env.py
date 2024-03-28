from enum import Enum
from numba.experimental import jitclass

class Action(Enum):
    FOLD = 0
    CALL = 1
    RAISE = 2
    FOLD

@jitclass
class PokerEnvironment:
    def __init__(self, num_players: int):
        # Initialize the environment variables here
        self.current_pot = 0
        self.current_bet = 0

    @property
    def min_raise(self):
        return 
    
    def reset(self):
        # Reset the environment to its initial state
        pass
    
    def step(self, action: Action):
        """_summary_

        Args:
            action (Action): _description_
        """
        # Perform the given action and return the next state, reward, and done flag
        pass
    
    def render(self):
        # Render the current state of the environment
        pass