import bisect
from collections import namedtuple, Counter, defaultdict

# import heapq
import numpy as np
import random

Card = namedtuple("Card", ["rank", "suit"])


class PokerEnvironment:
    def __init__(self, num_players=2, minimum_bet=2, stash_size=10):
        self.num_players = num_players
        self.init_stash_size = stash_size
        self.players_stash = np.array(
            [self.init_stash_size for _ in range(num_players)]
        )  # Initial stash for each player
        self.initial_player_stashes = self.players_stash.copy()
        self.active_players = []
        self.pot = 0
        self.deck = None
        self.community_cards = None
        self.current_bet = None
        self.active_player_bets = np.zeros(self.num_players)
        self.minimum_bet = minimum_bet
        self.bet_leader = None
        self.leader = 0
        self.hands = None
        self.current_player = None
        self.round = None

    def __get_community_cards(self):
        return self.community_cards[: {0: 0, 1: 3, 2: 4, 3: 5, 4: 5}[self.round]]

    def __get_player_state(self, player):
        return (
            self.__get_community_cards(),
            player,
            self.players_stash[player],
            self.hands[player],
        )

    def reset(self):
        self.reset_deck()
        self.hands = [
            [self.__get_card() for _ in range(2)] for _ in range(self.num_players)
        ]
        self.community_cards = [self.__get_card() for _ in range(5)]
        self.current_player = self.leader
        self.bet_leader = self.leader
        self.initial_player_stashes = self.players_stash.copy()
        self.active_players = [
            i for i in range(self.num_players) if self.players_stash[i] > 0
        ]
        self.current_bet = 0
        self.active_player_bets = np.zeros(self.num_players)
        # self.pot = 0
        self.round = 0  # Round 0 is before flop
        print(f"Community Cards: {self.community_cards}")
        print(f"Hands: {self.hands}")
        print(f"Best Hand(s): {self.__best_player_hand(range(self.num_players))}")
        return self.__get_player_state(self.current_player)

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

    def __stash_to_pot(self, player, amount):
        if self.players_stash[player] < amount:
            self.pot += self.players_stash[player]
            self.players_stash[player] = 0
        else:
            self.pot += amount
            self.active_player_bets[player] += amount
            self.players_stash[player] -= amount
            self.players_stash[player] -= amount

    def __pot_to_stash(self, players):
        split = self.pot // len(players)
        for player in players:
            self.players_stash[player] += split
            self.pot -= split

    def step(self, action):
        reward = 0
        done = False
        current_index = self.active_players.index(self.current_player)

        if self.round == 4:
            reward = self.players_stash[self.current_player] - self.initial_player_stashes[self.current_player]
        elif action == 0:  # Fold
            # print(f"Player {self.current_player} folded")
            self.active_players.remove(self.current_player)
            if self.current_player == self.bet_leader:
                if self.active_players:
                    self.bet_leader = self.active_players[current_index % len(self.active_players)]
                else:
                    self.bet_leader = None
        elif action == 1:  # Call
            self.__stash_to_pot(self.current_player, self.current_bet - self.active_player_bets[self.current_player])
        elif action == 2:  # Raise
            bet_raise = min(self.players_stash[self.current_player], max(self.minimum_bet, self.current_bet * 2))
            self.__stash_to_pot(self.current_player, bet_raise - self.active_player_bets[self.current_player])
            if bet_raise > self.current_bet:
                self.current_bet = bet_raise
                self.bet_leader = self.current_player
        elif action == 3:  # All-in
            all_in_amount = self.players_stash[self.current_player]
            self.__stash_to_pot(self.current_player, all_in_amount)
            if all_in_amount > self.current_bet:
                self.current_bet = all_in_amount
                self.bet_leader = self.current_player
        else:
            raise ValueError("Invalid action")

        # Update current player taking account of folded players
        if self.active_players:
            current_index = (current_index + 1) % len(self.active_players)
            self.current_player = self.active_players[current_index]

            if self.current_player == self.bet_leader:
                if self.round == 4:
                    done = True
                    if not self.game_over:
                        self.__reset_round()
                else:
                    if self.round == 3:
                        winners = self.__best_player_hand(self.active_players)
                        self.__pot_to_stash(winners)
                    # self.round += 1
                    self.__reset_bets()
        else:
            done = True  # All players except one have folded, so end the game

        return self.__get_player_state(self.current_player), reward, done, {}

    def __reset_round(self):
        self.leader = (self.leader + 1) % self.num_players
        while self.players_stash[self.leader] == 0 and any(self.players_stash):
            self.leader = (self.leader + 1) % self.num_players

    @property
    def game_over(self):
        return (self.round == 4 or self.round == 0) and sum(
            p for p in range(self.num_players) if self.players_stash[p] > 0
        ) <= 1

    def __reset_bets(self):
        self.current_bet = 0
        self.active_player_bets = np.zeros(self.num_players)
        if self.leader in self.active_players:
            self.bet_leader = self.leader
        else:
            self.bet_leader = self.active_players[
                (bisect.bisect(self.active_players, self.leader) + 1)
                % len(self.active_players)
            ]
        self.current_player = self.bet_leader
        self.round += 1

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
    env.render()
    done = False
    # Need DQN agent
    while not done:
        action = np.random.randint(0, 3)  # Random action for now
        state, reward, done, _ = env.step(action)
        env.render()
