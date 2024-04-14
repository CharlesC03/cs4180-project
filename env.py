import bisect
from collections import namedtuple, Counter, defaultdict

# import heapq
import numpy as np
import random

Card = namedtuple("Card", ["rank", "suit"])


class PokerEnvironment:
    def __init__(self, num_players=2, stash_size=10):
        self.num_players = num_players
        self.players_stash = np.array(
            [stash_size for _ in range(num_players)]
        )  # Initial stash for each player
        self.deck = None
        self.community_cards = None
        self.hands = None
        self.current_player = None
        self.round = None

    def __get_community_cards(self):
        return self.community_cards[: {0: 0, 1: 3, 2: 4, 3: 5}[self.round]]

    def __get_player_state(self):
        return (
            self.__get_community_cards(),
            self.current_player,
            self.players_stash[self.current_player],
            self.hands[self.current_player],
        )

    def reset(self):
        self.reset_deck()
        self.hands = [
            [self.__get_card() for _ in range(2)] for _ in range(self.num_players)
        ]
        self.community_cards = [self.__get_card() for _ in range(5)]
        self.current_player = 0
        self.round = 0  # Round 0 is before flop
        print(f"Community Cards: {self.community_cards}")
        print(f"Hands: {self.hands}")
        print(f"Best Hand(s): {self.__best_player_hand(range(self.num_players))}")
        return self.__get_player_state()

    def reset_deck(self):
        deck = [
            Card(rank, suit) for rank in range(2, 15) for suit in ["H", "D", "C", "S"]
        ]
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
        for i, card in enumerate(cards):
            if all(
                [
                    n in rank_info or (n == 1 and 14 in rank_info)
                    for n in range(card.rank, card.rank - 5, -1)
                ]
            ):
                s = [card]
                for c in cards[i:]:
                    if c.rank > card.rank - 5 and (
                        c not in s or c.rank not in [r.rank for r in s]
                    ):
                        s.append(c)
                if len(s) < 5 and 14 in rank_info:
                    s.append(
                        Card(
                            1,
                            card.suit
                            if Card(14, card.suit) in cards
                            else cards[0].suit,
                        )
                    )
                return s
        return []

    def __compare_hands(self, ref, hand):
        if len(ref) != len(hand):
            return 0b00
        for r, v in zip(ref, hand):
            if r > v:
                return 0b01
            if v > r:
                return 0b10
        return 0b11

    # Make this return as soon as we get a hand
    # Check hands in order of best to worst
    def __best_player_hand(self, players):
        best_players = []
        best_hand_rating = 9
        best_rank = None
        # hand ranking: Straight Flush: 0, Four of a Kind: 1, Full House: 2, Flush: 3, Straight: 4, Three of a Kind: 5, Two Pair: 6, One Pair: 7, High Card: 8
        for player in players:
            cards = self.hands[player] + self.community_cards
            cards = sorted(cards, key=lambda x: x.rank, reverse=True)

            suit_info = Counter([card.suit for card in cards])
            rank_info = Counter([card.rank for card in cards])
            # Get highest flush
            flush = self.__find_flush(cards, suit_info)
            straights = []
            if len(flush) > 0:
                straights = self.__find_straights(flush, rank_info)
            same_rank_info = defaultdict(list)
            {
                bisect.insort(same_rank_info[v], k, key=lambda x: -x)
                for k, v in rank_info.items()
            }
            check_func = {
                0: lambda: len(straights),  # straight flush
                1: lambda: len(same_rank_info[4]),  # four of a kind
                2: lambda: (  # full house
                    (len(same_rank_info[3]) and len(same_rank_info[2]))
                    or len(same_rank_info[3]) > 1
                ),
                3: lambda: len(flush),  # flush
                4: lambda: len(straights),  # straight
                5: lambda: len(same_rank_info[3]),  # three of a kind
                6: lambda: len(same_rank_info[2]) > 1,  # two pair
                7: lambda: len(same_rank_info[2]),  # one pair
                8: lambda: True,  # high card
            }
            get_info_func = {
                0: lambda: (straights[0].rank,),  # straight flush
                1: lambda: (  # four of a kind
                    same_rank_info[4][0],
                    max((rank for rank in rank_info if rank != same_rank_info[4][0])),
                ),
                2: lambda: (  # full house
                    same_rank_info[3][0],
                    max((same_rank_info[2][0], *same_rank_info[3][1:2])),
                ),
                3: lambda: flush[:5],  # flush
                4: lambda: (straights[0].rank,),  # straight
                5: lambda: (
                    same_rank_info[3][0],
                    *same_rank_info[1][:2],
                ),  # three of a kind
                6: lambda: (*same_rank_info[2][:2], same_rank_info[1][0]),  # two pair
                7: lambda: (same_rank_info[2][0], *same_rank_info[1][:3]),  # one pair
                8: lambda: same_rank_info[1][:5],  # high card
            }
            for level in range(best_hand_rating + 1):
                if check_func[level]():
                    ranking_info = get_info_func[level]()
                    if best_hand_rating > level:
                        best_hand_rating = level
                        best_rank = ranking_info
                        best_players = [player]
                    else:
                        better_hand = self.__compare_hands(best_rank, ranking_info)
                        if better_hand & 0b10:
                            if better_hand == 0b11:
                                best_players.append(player)
                            else:
                                best_rank = ranking_info
                                best_players = [player]
                    break
                if level == 3:
                    straights = self.__find_straights(cards, rank_info)
        # print(f"Best Hand: {best_hand_rating}, Best Rank: {best_rank}")
        return best_players

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
        print(
            f"Round: {self.round}, Current Player: {self.current_player}, Hands: {self.hands}, Community Cards: {self.__get_community_cards()}"
        )

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
