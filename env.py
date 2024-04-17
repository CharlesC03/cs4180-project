import bisect
from collections import namedtuple, Counter, defaultdict

# import heapq
import numpy as np
import random

Card = namedtuple("Card", ["rank", "suit"])


def cards_to_int(cards: list[Card]):
    return [1 + (card.rank - 2) * 4 + "HDCS".index(card.suit) for card in cards]


def cards_to_ints(cards: list[Card], length):
    res = [0] * (length * 2)
    for i, card in enumerate(cards):
        res[i * 2] = card.rank - 1
        res[i * 2 + 1] = "HDCS".index(card.suit) + 1
    return res


card_rank_str_mapping = {
    14: "A",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "T",
    11: "J",
    12: "Q",
    13: "K",
}


def card_to_str(card: Card):
    return f"{card.suit}{card_rank_str_mapping[card.rank]}"


class PokerEnvironment:
    """The PokerEnvironment class implements a simplified version of the Texas Hold'em poker game."""

    def __init__(
        self, num_players=2, minimum_bet=0.5, stash_size=10, little_blind=False
    ):
        """
        Initializes an instance of the PokerEnvironment class.

        Args:
            num_players (int): The number of players in the game. Default is 2.
            minimum_bet (float): The minimum bet amount. Default is 0.5.
            stash_size (float): The initial stash size for each player. Default is 10.
            little_blind (bool): Indicates whether the game uses a little blind. Default is False.
        """
        self.num_players = num_players
        self.init_stash_size = stash_size
        self.players_stash = np.array(
            [self.init_stash_size for _ in range(num_players)], dtype=np.float64
        )  # Initial stash for each player
        self.initial_player_stashes = self.players_stash.copy()
        self.active_players = []
        self.pot = 0
        self.deck = None
        self.community_cards = None
        self.current_bet = None
        self.active_player_bets = np.zeros(self.num_players)
        self.minimum_bet = minimum_bet
        self.little_blind = little_blind
        self.bet_leader = None
        self.leader = 0
        self.hands = None
        self.current_player = None
        self.round = None
        self.next_player = -1
        self.action_space = 4

    def __get_community_cards(self):
        """
        Returns a subset of the community cards based on the current round.

        The number of community cards returned depends on the current round of the game.
        - In round 0, no community cards are returned.
        - In round 1, 3 community cards are returned.
        - In round 2, 4 community cards are returned.
        - In rounds 3 and 4, 5 community cards are returned.

        Returns:
            list: A subset of the community cards based on the current round.
        """
        return self.community_cards[: {0: 0, 1: 3, 2: 4, 3: 5, 4: 5}[self.round]]

    @property
    def state_shape(self):
        # return 16 + self.num_players
        return len(self.__get_player_state(0))

    def __get_player_state(self, player):
        """
        Get the state of a player.

        Args:
            player (str): The name of the player.

        Returns:
            tuple: A tuple containing the player's name, the community cards, the player's stash, and the player's hand.
        """
        return (
            (self.bet_leader - player + self.num_players) % self.num_players,
            *cards_to_ints(self.__get_community_cards(), 5),
            *cards_to_ints(self.hands[player], 2),
            *self.players_stash,
            self.current_bet - self.active_player_bets[player],
        )

    def reset(self):
        """
        Resets the game state to its initial state.

        This method resets the deck, hands, community cards, current player, bet leader,
        initial player stashes, active players, current bet, active player bets, and round.
        It returns the player state of the current player.

        Returns:
            int: The name of the current player.
            dict: The player state of the current player.
        """
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
        self.next_player = -1
        self.current_bet = 0
        self.active_player_bets = np.zeros(self.num_players)
        self.round = 0  # Round 0 is before flop
        return self.current_player, self.__get_player_state(self.current_player)

    def reset_deck(self):
        """
        Resets the deck of cards by creating a new deck, shuffling it, and assigning it to the instance variable 'deck'.
        """
        deck = [
            Card(rank, suit) for rank in range(2, 15) for suit in ["H", "D", "C", "S"]
        ]
        random.shuffle(deck)
        self.deck = deck

    def __find_flush(self, cards: list[Card], suit_info: Counter) -> list[Card]:
        """
        Find and return a list of cards that form a flush.

        Args:
            cards (list[Card]): The list of cards to check for a flush.
            suit_info (Counter): A Counter object containing the count of each suit in the cards.

        Returns:
            list[Card]: A list of cards that form a flush, or an empty list if no flush is found.
        """
        if max(suit_info.values()) >= 5:
            return [card for card in cards if suit_info[card.suit] >= 5]
        return []

    def __find_straights(self, cards: list[Card], rank_info: Counter) -> list[Card]:
        """
        Find and return a list of straight cards from the given list of cards.

        Args:
            cards (list[Card]): The list of cards to search for straights.
            rank_info (Counter): A Counter object containing the rank information of the cards.

        Returns:
            list[Card]: A list of straight cards found, or an empty list if no straight is found.
        """
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
        """
        Compares two hands of cards and returns a binary value indicating the comparison result.

        Args:
            ref (list): The reference hand of cards.
            hand (list): The hand of cards to compare against the reference hand.

        Returns:
            int: A binary value indicating the comparison result. The binary value is represented as an integer.
                - If the lengths of the two hands are not equal, returns 0b00.
                - If the reference hand is greater than the compared hand, returns 0b01.
                - If the compared hand is greater than the reference hand, returns 0b10.
                - If both hands are equal, returns 0b11.
        """
        if len(ref) != len(hand):
            return 0b00
        for r, v in zip(ref, hand):
            if r > v:
                return 0b01
            if v > r:
                return 0b10
        return 0b11

    def __best_player_hand(self, players):
        """Given a list of players, finds the players with the best hand.

        Args:
            players (list[int]): List of player numbers to check for the best hand.

        Returns:
            list[int]: List of player numbers with the best hand.
        """
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
                    max((*same_rank_info[2][0:1], *same_rank_info[3][1:2])),
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
        """
        Moves chips from a player's stash to the pot.

        Args:
            player (str): The name of the player.
            amount (int): The amount of chips to move to the pot.

        Raises:
            ValueError: If the amount is negative.

        Returns:
            None
        """
        if amount < 0:
            raise ValueError("Amount must be positive")
        if self.players_stash[player] < amount:
            amount = self.players_stash[player]
        self.pot += amount
        self.active_player_bets[player] += amount
        self.players_stash[player] -= amount
        if self.players_stash[player] == 0:
            self.__fold_player(player)

    def __pot_to_stash(self, players):
        """
        Distributes the pot equally among the given players.

        Args:
            players (list): A list of players to distribute the pot to.

        Returns:
            None
        """
        split = self.pot / len(players)
        for player in players:
            self.players_stash[player] += split
        self.pot = 0
        self.active_player_bets = np.zeros(self.num_players)
        self.current_bet = 0

    def __get_next_player(self):
        """
        Returns the next player in the list of all players.

        Returns:
            The next player in the list of all players.
        """
        return (self.current_player + 1) % self.num_players

    def __get_last_player(self):
        """
        Returns the previous player in the list of all players.

        Returns:
            The previous player in the list of all players.
        """
        return (self.current_player - 1) % self.num_players

    def __get_next_active_player(self):
        """
        Returns the next player in the list of active players.

        The next player is determined by finding the index of the current player
        in the list of active players and then adding 1 to it. If the resulting index
        exceeds the length of the list, the modulo operator is used to wrap around
        to the beginning of the list.

        Returns:
            The next player in the list of active players.
        """
        return self.active_players[
            bisect.bisect(self.active_players, self.current_player)
            % len(self.active_players)
        ]

    def __get_last_active_player(self):
        """
        Returns the player who played before the current player.

        Returns:
            The player who played before the current player.
        """
        return self.active_players[
            (
                len(self.active_players)
                + bisect.bisect(self.active_players, self.current_player)
                - 2
            )
            % len(self.active_players)
        ]

    def __get_player_reward(self, player):
        """
        Calculate the reward for a given player.

        Parameters:
        player (str): The name of the player.

        Returns:
        int: The reward for the player, calculated as the difference between the player's current stash and their initial stash.
        """
        return self.players_stash[player] - self.initial_player_stashes[player]

    def __fold_player(self, player):
        """
        Removes the specified player from the active players list and updates the current player.

        Parameters:
        - player: The player to be folded.
        """
        if player in self.active_players and len(self.active_players) == 1:
            raise ValueError("Cannot fold last active player")
        if self.current_player == player:
            self.next_player = self.__get_next_active_player()
        if self.bet_leader == player:
            self.bet_leader = self.__get_next_active_player()
        if player in self.active_players:
            self.active_players.remove(player)

    def __call_player(self, player):
        """
        Calls the specified player in the game.

        Parameters:
        player (str): The name of the player to be called.

        Returns:
        None
        """
        self.__stash_to_pot(player, self.current_bet - self.active_player_bets[player])

    def __raise_player(self, player, amount=None):
        """
        Raises the bet for a given player.

        Args:
            player (str): The name of the player.
            amount (int, optional): The amount to raise the bet by. If not provided, the bet is raised to either the minimum bet or double the current bet.

        Returns:
            None
        """
        # set the amount to raise to, with either the minimum bet or double the current bet (if not enough money goes all in)
        new_bet = self.current_bet + min(
            self.players_stash[player],
            max(self.minimum_bet, self.current_bet if amount is None else amount),
        )
        # add the raise to the pot
        self.__stash_to_pot(
            player,
            new_bet - self.active_player_bets[player],
        )
        # if the raise is greater than the current bet, set the current bet to the raise and the bet leader to the player
        if new_bet > self.current_bet:
            self.current_bet = new_bet
            self.bet_leader = player

    def __get_players_rewards(self):
        """
        Returns a dictionary containing the rewards for each player.

        Returns:
            dict: A dictionary where the keys are player numbers and the values are their respective rewards.
        """
        return {
            player: self.__get_player_reward(player)
            for player in range(self.num_players)
        }

    def step(self, action):
        """
        Perform a step in the game environment.

        Args:
            action (int): The action to take in the game.

        Returns:
            tuple: A tuple containing the current player, the player state, the player reward, and a boolean indicating if the game is done.

        Raises:
            ValueError: If the current player is not active or if an invalid action is provided.
        """
        if self.round == 4:
            self.current_player = self.__get_next_player()
            done = False
            if self.current_player == self.leader:
                if not self.game_over:
                    self.__set_next_leader()
                done = True
            return (
                self.current_player,
                self.__get_player_state(self.current_player),
                self.__get_player_reward(self.current_player),
                done,
            )

        if (
            len(self.active_players) == 1
            or sum([self.players_stash[p] != 0 for p in self.active_players]) == 1
        ):
            # No active players remaining (all have folded or game is over)
            self.__pot_to_stash(self.active_players)
            self.current_player = self.leader
            self.current_player = self.__get_next_player()
            self.round = 4
            return (
                self.current_player,
                self.__get_player_state(self.current_player),
                self.__get_player_reward(self.current_player),
                False,
            )

        # Ensure that self.current_player is still in active_players
        if self.current_player not in self.active_players:
            raise ValueError("Current player is not active")

        if self.round == 0 and self.bet_leader == self.current_player:
            self.current_bet = self.minimum_bet
            self.__call_player(self.current_player)
        elif (
            self.little_blind
            and self.round == 0
            and self.bet_leader == self.__get_last_active_player()
        ):
            little_blind = self.minimum_bet / 2
            self.__stash_to_pot(self.current_player, little_blind)

        if action == 0:  # Fold
            # Player folds and is removed from active players
            self.__fold_player(self.current_player)
        elif action == 1:  # Call
            self.__call_player(self.current_player)
        elif action == 2:  # Raise
            self.__raise_player(self.current_player)
        elif action == 3:  # All-in
            self.__raise_player(
                self.current_player, self.players_stash[self.current_player]
            )
        else:
            raise ValueError(f"Invalid action: {action}")

        # Perform other actions based on the chosen action (Call, Raise, All-in)
        # Move to the next active player
        self.current_player = (
            self.next_player
            if self.next_player != -1
            else self.__get_next_active_player()
        )

        # Check if the betting round or game is over
        if self.current_player == self.bet_leader:
            if self.round == 4:
                if not self.game_over:
                    self.__set_next_leader()
            elif self.round == 3:
                winners = self.__best_player_hand(self.active_players)
                self.__pot_to_stash(winners)
                self.current_player = self.leader
                self.current_player = self.__get_next_player()
                self.round = 4
                return (
                    self.current_player,
                    self.__get_player_state(self.current_player),
                    self.__get_player_reward(self.current_player),
                    False,
                )
            self.__reset_bets()

        return (
            self.current_player,
            self.__get_player_state(self.current_player),
            0,
            False,
        )

    def __set_next_leader(self):
        self.leader = (self.leader + 1) % self.num_players
        while self.players_stash[self.leader] == 0:
            self.leader = (self.leader + 1) % self.num_players

    @property
    def game_over(self):
        """
        Check if the game is over.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        return (
            self.pot == 0
            and sum(1 for p in range(self.num_players) if self.players_stash[p] > 0)
            <= 1
        )

    def __reset_bets(self):
        """
        Resets the bets for a new round of the game.

        This method resets the current bet, active player bets, bet leader, current player, and round
        attributes to their initial values for a new round of the game.

        Parameters:
            None

        Returns:
            None
        """
        self.current_bet = 0
        self.active_player_bets = np.zeros(self.num_players)
        if self.leader in self.active_players:
            self.bet_leader = self.leader
        else:
            self.bet_leader = self.active_players[
                bisect.bisect(self.active_players, self.leader)
                % len(self.active_players)
            ]
        self.current_player = self.bet_leader
        self.round += 1

    def render(self, action="N/A"):
        """
        Prints the current state of the game.

        If the round is 3 or there is only one active player remaining, it prints the winners, pot, rewards, and new leader.
        Otherwise, it prints the round number, current player, stash, hands, and community cards.
        """
        print(
            f"Round: {self.round}, Current Player: {self.current_player}, Pot:{self.pot}, Stashes: {[f'{player}: {self.players_stash[player]}' for player in range(self.num_players)]}, Hand: {', '.join([card_to_str(card) for card in self.hands[self.current_player]])}, Community Cards: {[f'{card.suit}{card.rank}' for card in self.__get_community_cards()]}"
        )
        if self.round == 4 or len(self.active_players) == 1:
            print(
                f"Winners: {self.__best_player_hand(self.active_players)}, Pot: {self.pot}, Rewards: {self.__get_players_rewards()}, New Leader: {self.leader}"
            )

    def random_action(self):
        return np.random.randint(0, 4)

    def close(self):
        pass

    def full_reset(self):
        self.__init__(
            self.num_players, self.minimum_bet, self.init_stash_size, self.little_blind
        )


# # To run
if __name__ == "__main__":
    # for _ in range(int(1e6)):
    env = PokerEnvironment(2)
    while not env.game_over:
        state = env.reset()
        done = False
        # Need DQN agent
        while not done:
            action = env.random_action()  # Random action for now
            print(f"Action: {action}")
            player, state, reward, done = env.step(action)
            print(
                f"Round: {env.round}, Action: {action}, Player: {player}, State: {state}, Reward: {reward}, Done: {done}"
            )
