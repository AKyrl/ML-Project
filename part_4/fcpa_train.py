from random import shuffle
from math import ceil
from collections import Counter
from itertools import combinations
import pandas as pd


def chen(hand):
    ranks = "23456789TJQKA"
    facePoints = {"A": 10, "K": 8, "Q": 7, "J": 6, "T": 5}

    if ranks.index(hand[0][0]) < ranks.index(hand[1][0]):  # Sort hand:
        hand[0], hand[1] = hand[1], hand[0]

    card1, card2 = hand[0], hand[1]
    rank1, rank2 = card1[0], card2[0]
    suit1, suit2 = card1[1], card2[1]
    # Score highest card
    if rank1 in facePoints:
        score = facePoints.get(rank1)
    else:
        score = int(rank1) / 2.

    # Multiply pairs by 2 of one card's value
    if rank1 is rank2:
        score *= 2
        if score < 5:
            score = 5

    # Add 2 if cards are suited
    if suit1 is suit2:
        score += 2

    # Subtract points if there is a gap
    gap = ranks.index(rank1) - ranks.index(rank2) - 1
    gapPoints = {1: 1, 2: 2, 3: 4}
    if gap in gapPoints:
        score -= gapPoints.get(gap)
    elif gap >= 4:
        score -= 5

    # Straight bonus
    if (gap < 2) and (ranks.index(rank1) < ranks.index("Q")) and (rank1 is not rank2):
        score += 1

    return int(ceil(score))


# gets the most common element from a list
def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0]


# gets card value from  a hand. converts A to 14,  is_seq function will convert the 14 to a 1 when necessary to evaluate A 2 3 4 5 straights
def convert_tonums(h, nums={'T': 10, 'J': 11, 'Q': 12, 'K': 13, "A": 14}):
    for x in range(len(h)):
        if (h[x][0]) in nums.keys():
            h[x] = str(nums[h[x][0]]) + h[x][1]
    return h


# is royal flush
# if a hand is a straight and a flush and the lowest value is a 10 then it is a royal flush
def is_royal(h):
    if len(h) >= 5:
        nh = convert_tonums(h)
        if is_seq(h):
            if is_flush(h):
                nn = [int(x[:-1]) for x in nh]
                if min(nn) == 10:
                    return True

        else:
            return False


# converts hand to number valeus and then evaluates if they are sequential  AKA a straight
def is_seq(h):
    if len(h) >= 5:
        return is_seq_not5(h)


def is_seq_not5(h):
    ace = False
    r = h[:]

    h = [x[:-1] for x in convert_tonums(h)]

    h = [int(x) for x in h]
    h = list(sorted(h))
    ref = True
    for x in range(0, len(h) - 1):
        if not h[x] + 1 == h[x + 1]:
            ref = False
            break

    if ref:
        return True, r

    aces = [i for i in h if str(i) == "14"]
    if len(aces) == 1:
        for x in range(len(h)):
            if str(h[x]) == "14":
                h[x] = 1

    h = list(sorted(h))
    for x in range(0, len(h) - 1):
        if not h[x] + 1 == h[x + 1]:
            return False
    return True, r


def is_potential_seq(h):
    if len(h) == 2:
        if is_seq_not5(h):
            return 10
        else:
            return 0
    elif len(h) == 5:
        comb = combinations(h, 4)
        for combination in list(comb):
            if is_seq_not5(list(combination)):
                return 30
        comb = combinations(h, 3)
        for combination in list(comb):
            if is_seq_not5(list(combination)):
                return 20
        return 0
    elif len(h) == 6:
        comb = combinations(h, 4)
        for combination in list(comb):
            if is_seq_not5(list(combination)):
                return 20
        return 0
    else:
        return 0


# call set() on the suite values of the hand and if it is 1 then they are all the same suit
def is_flush(h):
    if len(h) >= 5:
        suits = [x[-1] for x in h]
        if len(set(suits)) == 1:
            return True, h
        else:
            return False


def is_potential_flush(h):
    if len(h) == 2:
        suits = [x[-1] for x in h]
        if len(set(suits)) == 1:
            return 20
        else:
            return 0
    elif len(h) == 5:
        comb = combinations(h, 4)
        for combination in list(comb):
            suits = [x[-1] for x in list(combination)]
            if len(set(suits)) == 1:
                return 50
        comb = combinations(h, 3)
        for combination in list(comb):
            suits = [x[-1] for x in list(combination)]
            if len(set(suits)) == 1:
                return 40
        return 0
    elif len(h) == 6:
        comb = combinations(h, 4)
        for combination in list(comb):
            suits = [x[-1] for x in list(combination)]
            if len(set(suits)) == 1:
                return 40
        return 0
    else:
        return 0


# if the most common element occurs 4 times then it is a four of a kind
def is_fourofakind(h):
    if len(h) >= 5:
        h = [a[:-1] for a in h]
        i = Most_Common(h)
        if i[1] == 4:
            return True, i[0]
        else:
            return False


# if the most common element occurs 3 times then it is a three of a kind
def is_threeofakind(h):
    if len(h) >= 5:
        h = [a[:-1] for a in h]
        i = Most_Common(h)
        if i[1] == 3:
            return True, i[0]
        else:
            return False


# if the first 2 most common elements have counts of 3 and 2, then it is a full house
def is_fullhouse(h):
    if len(h) >= 5:
        h = [a[:-1] for a in h]
        data = Counter(h)
        a, b = data.most_common(1)[0], data.most_common(2)[-1]
        if str(a[1]) == '3' and str(b[1]) == '2':
            return True, (a, b)
        return False


# if the first 2 most common elements have counts of 2 and 2 then it is a two pair
def is_twopair(h):
    if len(h) >= 5:
        h = [a[:-1] for a in h]
        data = Counter(h)
        a, b = data.most_common(1)[0], data.most_common(2)[-1]
        if str(a[1]) == '2' and str(b[1]) == '2':
            return True, (a[0], b[0])
        return False


# if the first most common element is 2 then it is a pair
# DISCLAIMER: this will return true if the hand is a two pair, but this should not be a conflict because is_twopair is always evaluated and returned first
def is_pair(h):
    h = [a[:-1] for a in h]
    data = Counter(h)
    a = data.most_common(1)[0]

    if str(a[1]) == '2':
        return True, (a[0])
    else:
        return False


# get the high card
def get_high(h):
    return list(sorted([int(x[:-1]) for x in convert_tonums(h)], reverse=True))[0]


# FOR HIGH CARD or ties, this function compares two hands by ordering the hands from highest to lowest and comparing each card and returning when one is higher then the other
def compare(xs, ys):
    xs, ys = list(sorted(xs, reverse=True)), list(sorted(ys, reverse=True))
    for i, c in enumerate(xs):
        if ys[i] > c:
            return 'RIGHT'
        elif ys[i] < c:
            return 'LEFT'
    return "TIE"


# categorized a hand based on previous functions
def evaluate_hand(h):
    if is_royal(h):
        return "ROYAL FLUSH", h, 370
    elif is_seq(h) and is_flush(h):
        return "STRAIGHT FLUSH", h, 330
    elif is_fourofakind(h):
        _, fourofakind = is_fourofakind(h)
        return "FOUR OF A KIND", fourofakind, 290
    elif is_fullhouse(h):
        return "FULL HOUSE", h, 250
    elif is_flush(h):
        _, flush = is_flush(h)
        return "FLUSH", h, 210
    elif is_seq(h):
        _, seq = is_seq(h)
        return "STRAIGHT", h, 170
    elif is_threeofakind(h):
        _, threeofakind = is_threeofakind(h)
        return "THREE OF A KIND", threeofakind, 130
    elif is_twopair(h):
        _, two_pair = is_twopair(h)
        return "TWO PAIR", two_pair, 90
    elif is_pair(h):
        _, pair = is_pair(h)
        return "PAIR", pair, 50
    else:
        return "HIGH CARD", h, 0


# this monster function evaluates two hands and also deals with ties and edge cases
# this monster function evaluates two hands and also deals with ties and edge cases
# this probably should be broken up into separate functions but aint no body got time for that
def compare_hands(h1, h2):
    one, two = evaluate_hand(h1), evaluate_hand(h2)
    if one[0] == two[0]:
        if one[0] == "STRAIGHT FLUSH":
            sett1, sett2 = convert_tonums(h1), convert_tonums(h2)
            sett1, sett2 = [int(x[:-1]) for x in sett1], [int(x[:-1]) for x in sett2]
            com = compare(sett1, sett2)
            if com == "TIE":
                return "none", one[1], two[1]
            elif com == "RIGHT":
                return "right", two[0], two[1]
            else:
                return "left", one[0], one[1]

        elif one[0] == "TWO PAIR":
            leftover1, leftover2 = is_twopair(h1), is_twopair(h2)
            twm1, twm2 = max([int(x) for x in list(leftover1[1])]), max([int(x) for x in list(leftover2[1])])
            if twm1 > twm2:
                return "left", one[0], one[1]
            elif twm1 < twm2:
                return "right", two[0], two[1]

            if compare(list(leftover1[1]), list(leftover2[1])) == "TIE":
                l1 = [x[:-1] for x in h1 if x[:-1] not in leftover1[1]]
                l2 = [x[:-1] for x in h2 if x[:-1] not in leftover2[1]]
                if int(l1[0]) == int(l2[0]):
                    return "none", one[1], two[1]
                elif int(l1[0]) > int(l2[0]):
                    return "left", one[0], one[1]
                else:
                    return "right", two[0], two[1]
            elif compare(list(leftover1[1]), list(leftover2[1])) == "RIGHT":
                return "right", two[0], two[1]
            elif compare(list(leftover1[1]), list(leftover2[1])) == "LEFT":
                return "left", one[0], one[1]

        elif one[0] == "PAIR":
            sh1, sh2 = int(is_pair(h1)[1]), int(is_pair(h2)[1])
            if sh1 == sh2:

                c1 = [int(x[:-1]) for x in convert_tonums(h1) if not int(sh1) == int(x[:-1])]
                c2 = [int(x[:-1]) for x in convert_tonums(h2) if not int(sh1) == int(x[:-1])]
                if compare(c1, c2) == "TIE":
                    return "none", one[1], two[1]
                elif compare(c1, c2) == "RIGHT":
                    return "right", two[0], two[1]
                else:
                    return "left", one[0], one[1]

            elif h1 > h2:
                return "right", two[0], two[1]
            else:
                return "left", one[0], one[1]

        elif one[0] == 'FULL HOUSE':

            fh1, fh2 = int(is_fullhouse(h1)[1][0][0]), int(is_fullhouse(h2)[1][0][0])
            if fh1 > fh2:
                return "left", one[0], one[1]
            else:
                return "right", two[0], two[1]
        elif one[0] == "HIGH CARD":
            sett1, sett2 = convert_tonums(h1), convert_tonums(h2)
            sett1, sett2 = [int(x[:-1]) for x in sett1], [int(x[:-1]) for x in sett2]
            com = compare(sett1, sett2)
            if com == "TIE":
                return "none", one[1], two[1]
            elif com == "RIGHT":
                return "right", two[0], two[1]
            else:
                return "left", one[0], one[1]



        elif len(one[1]) < 5:
            if max(one[1]) == max(two[1]):
                return "none", one[1], two[1]
            elif max(one[1]) > max(two[1]):
                return "left", one[0], one[1]
            else:
                return "right", two[0], two[1]
        else:
            n_one, n_two = convert_tonums(one[1]), convert_tonums(two[1])
            n_one, n_two = [int(x[:-1]) for x in n_one], [int(x[:-1]) for x in n_two]

            if max(n_one) == max(n_two):
                return "none", one[1], two[1]
            elif max(n_one) > max(n_two):
                return "left", one[0], one[1]
            else:
                return "right", two[0], two[1]
    elif one[2] > two[2]:
        return "left", one[0], one[1]
    else:
        return "right", two[0], two[1]


def best5(cards):
    if len(cards) < 5:
        return cards
    comb = combinations(cards, 5)
    best = cards[:5]
    counter = 0
    for combination in list(comb):
        comp = compare_hands(best, list(combination))
        counter += 1
        if comp[0] == "right":
            best = list(combination)
    return best


def individualPoints(cards):
    points = 0
    for rank in [card[0] for card in cards]:
        if rank in ['T', 'J', 'Q', 'K', 'A']:
            points += high_rank_to_int(rank)
        else:
            points += int(rank)
    return points


def evaluate(cards):
    if len(cards) == 5 or len(cards) == 6 or len(cards) == 2:
        points = individualPoints(cards) + evaluate_hand(cards)[2] + is_potential_seq(cards) + is_potential_flush(cards)
    else:
        cards = best5(cards)
        points = individualPoints(cards) + evaluate_hand(cards)[2]
    return points


def high_rank_to_int(high_rank):
    dictionary = {'T': 10, 'J': 11, 'Q': 12, 'K': 13, "A": 14}
    if high_rank not in dictionary.keys():
        return 0
    else:
        return dictionary[high_rank]


def cards_list_to_string(cards_list):
    cards = ""
    for card in cards_list:
        cards += card
    return cards


def generate_examples_5_cards(steps):
    ranks = "23456789TJQKA"
    suits = "cdhs"
    deck = [(j + i) for i in suits for j in ranks]
    shuffle(deck)
    examples = []
    for _ in range(steps):
        for i in range(7):
            player1_cards = [deck[5 * i + 0], deck[5 * i + 1]]
            player2_cards = [deck[5 * i + 2], deck[5 * i + 3]]
            board_cards = [deck[5 * i + 4], deck[5 * i + 5], deck[5 * i + 6]]
            chen1 = chen(player1_cards)
            chen2 = chen(player2_cards)
            heuristic1 = evaluate(player1_cards + board_cards)
            heuristic2 = evaluate(player2_cards + board_cards)
            res = compare_hands(player1_cards + board_cards, player2_cards + board_cards)
            if res[0] == "left":
                p1won = 1
                p2won = 0
            else:
                p1won = 0
                p2won = 1
            player1_cards = cards_list_to_string(player1_cards)
            player2_cards = cards_list_to_string(player2_cards)
            board_cards = cards_list_to_string(board_cards)
            examples.append([player1_cards, board_cards, chen1, heuristic1, p1won])
            examples.append([player2_cards, board_cards, chen2, heuristic2, p2won])
        shuffle(deck)

    df = pd.DataFrame(examples)
    return df


def generate_examples_2_cards(steps):
    ranks = "23456789TJQKA"
    suits = "cdhs"
    deck = [(j + i) for i in suits for j in ranks]
    shuffle(deck)
    examples = []
    for _ in range(steps):
        for i in range(13):
            player1_cards = [deck[4 * i + 0], deck[4 * i + 1]]
            player2_cards = [deck[4 * i + 2], deck[4 * i + 3]]
            chen1 = chen(player1_cards)
            chen2 = chen(player2_cards)
            heuristic1 = evaluate(player1_cards)
            heuristic2 = evaluate(player2_cards)
            res = compare_hands(player1_cards, player2_cards)
            if res[0] == "left":
                p1won = 1
                p2won = 0
            else:
                p1won = 0
                p2won = 1
            player1_cards = cards_list_to_string(player1_cards)
            player2_cards = cards_list_to_string(player2_cards)
            examples.append([player1_cards, chen1, heuristic1, p1won])
            examples.append([player2_cards, chen2, heuristic2, p2won])
        shuffle(deck)

    df = pd.DataFrame(examples)
    return df


def generate_examples_6_cards(steps):
    ranks = "23456789TJQKA"
    suits = "cdhs"
    deck = [(j + i) for i in suits for j in ranks]
    shuffle(deck)
    examples = []
    for _ in range(steps):
        for i in range(6):
            player1_cards = [deck[8 * i + 0], deck[8 * i + 1]]
            player2_cards = [deck[8 * i + 2], deck[8 * i + 3]]
            board_cards = [deck[8 * i + 4], deck[8 * i + 5], deck[8 * i + 6], deck[8 * i + 7]]
            chen1 = chen(player1_cards)
            chen2 = chen(player2_cards)
            player1_cards = best5(player1_cards + board_cards)
            player2_cards = best5(player2_cards + board_cards)
            heuristic1 = evaluate(player1_cards)
            heuristic2 = evaluate(player2_cards)
            res = compare_hands(player1_cards, player2_cards)
            if res[0] == "left":
                p1won = 1
                p2won = 0
            else:
                p1won = 0
                p2won = 1
            player1_cards = cards_list_to_string(player1_cards)
            player2_cards = cards_list_to_string(player2_cards)
            examples.append([player1_cards, "", chen1, heuristic1, p1won])
            examples.append([player2_cards, "", chen2, heuristic2, p2won])
        shuffle(deck)

    df = pd.DataFrame(examples)
    return df


# a = ['Qd', 'Kd', 'Jh', '4h', 'Qs', '2h']
# b = ['Js', '8s', 'Ks', '6d', 'Ah', '4s']
# # a = best5(a)
# # b = best5(b)
# # print(compare_hands(a, b))
# evaluate(a)


def main():
    df = generate_examples_5_cards(10)  # steps * 14
    df.to_csv("dataset5cards.csv")

    df = generate_examples_2_cards(10)  # steps * 13
    df.to_csv("dataset2cards.csv")


    # df = generate_examples_6_cards(10) # steps * 12
    # df.to_csv("dataset6cards.csv")


if __name__ == '__main__':
    main()
