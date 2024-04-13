from collections import namedtuple, Counter
import numpy as np
import random

Card = namedtuple('Card', ['rank', 'suit'])

class PokerEnvironment:
    def __init__(self, num_players=2, stash_size=10):
        self.num_players = num_players
        self.players_stash = np.array([stash_size for _ in range(num_players)])  # Initial stash for each player
        self.deck = None
        self.community_cards = None
        self.hands = None
        self.current_player = None
        self.round = None
        
    def __get_community_cards(self):
        return self.community_cards[:{0:0, 1:3, 2:4, 3:5}[self.round]]
    
    def __get_player_state(self):
        return self.__get_community_cards(), self.current_player, self.players_stash[self.current_player], self.hands[self.current_player]

    def reset(self):
        self.reset_deck()
        self.hands = [[self.__get_card() for _ in range(2)] for _ in range(self.num_players)]
        self.community_cards = [self.__get_card() for _ in range(5)]
        self.current_player = 0
        self.round = 0  # Round 0 is before flop
        print(self.__best_player_hand())
        return self.__get_player_state()
    
    def reset_deck(self):
        deck = [Card(rank, suit) for rank in range(1,13) for suit in ['H', 'D', 'C', 'S']]
        random.shuffle(deck)
        self.deck = deck
    
    # Returns the flush if it exists
    def __find_flush(self, cards: list[Card], suit_info: Counter) -> list[Card]:
        if max(suit_info.values()) >= 5:
            return [card for card in cards if suit_info[card.suit] >= 5]
        return []
    
    # Returns the straight if it exists
    def __find_straights(self, cards: list[Card], rank_info: Counter) -> list[Card]:
        if len(rank_info) < 5:
            return []
        straights = []
        for i, card in enumerate(cards):
            if all([n in rank_info for n in range(card.rank, card.rank-5, -1)]):
                straights.append(cards[i:i+5])
        return straights
        
    def __find_straight_flush(self, flush: list[Card]) -> list[Card]:
        if len(flush) == 0:
            return []
        rank_info = Counter([card.rank for card in flush])
        return self.__find_straights(flush, rank_info)
    
    # Make this return as soon as we get a hand
    # Check hands in order of best to worst
    def __best_player_hand(self):
        cards = [Card(1, 'D'), Card(13, 'D'), Card(11, 'D'), Card(12, 'D'), Card(10, 'D'), Card(9, 'D'), Card(8, 'D'), Card(7, 'S'), Card(6, 'S'), Card(5, 'C')]
        cards = sorted(cards, key=lambda x: x.rank, reverse=True)
        
        suit_info = Counter([card.suit for card in cards])
        rank_info = Counter([card.rank for card in cards])
        rank_info[14] = rank_info[1]
        # Get highest flush
        # flush = {suit: suit_info[suit] for suit in suit_info if suit_info[suit] >= 5}
        # flush =  if len(flush) > 0 else 0
        flush = self.__find_flush(cards, suit_info)
        straights = self.__find_straights(flush, rank_info)
        if len(straights) == 0:
            straights = self.__find_straights(cards, rank_info)
        print(f"Cards: {cards}")
        print(f"Straights: {straights}")
        # print(f"Straight: {straight}")
        print(f"Flush: {flush}")
    
    def __get_card(self):
        return self.deck.pop()

    def get_current_player_hand(self, player):
        return self.hands[player]

    def step(self, action):
        # Action: 0 for fold, 1 for call, 2 for raise, 3 for all-in
        # For simplicity, action space is 0 for fold, 1 for call/check/raise
        if action == 0:
            # Fold
            reward = -1  # Penalize folding
            done = True
        else:
            # Call/Check/Raise - For simplicity, consider this action as 'call' for now
            reward = 0  # No reward or penalty for calling
            done = False

        # Move to the next player
        self.current_player = (self.current_player + 1) % self.num_players

        # If all players have made their decision, proceed to the next round or end the game
        if self.current_player == 0:
            self.round += 1
            if self.round == 4:  # End of game
                done = True

        # State Representation (? for DQN, but not useful right now)
        state = np.zeros((1, 1))

        return state, reward, done, {}

    def render(self):
        print(f"Round: {self.round}, Current Player: {self.current_player}, Hands: {self.hands}, Community Cards: {self.__get_community_cards()}")

    def close(self):
        pass

# To run
if __name__ == "__main__":
    env = PokerEnvironment()
    state = env.reset()
    # env.render()
    # done = False
    # # Need DQN agent
    # while not done:
    #     action = np.random.randint(0, 2)  # Random action for now
    #     state, reward, done, _ = env.step(action)
    #     env.render()
