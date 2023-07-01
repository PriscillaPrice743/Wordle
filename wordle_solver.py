import numpy as np
import pandas as pd
import itertools


from typing import Tuple


# Constants.
G_UNICODE = ord("g")
Y_UNICODE = ord("y")
B_UNICODE = ord("b")


def main() -> None:
    """Run the Wordle Solver program."""
    words = get_all_valid_words()

    print("----------WORDLE SOLVER----------")

    for turn in range(6):
        if (turn == 0):
            # Use pre-computed (it just takes a while so why not cache it?) first word stats.
            first_word_stats = pd.read_csv("first_word_stats.csv")
            sorted_words = np.array([get_unicode(word) for word in first_word_stats["word"]])
            predicted_remaining_words = first_word_stats["predicted_remaining_words"].values
        else:
            sorted_words, predicted_remaining_words = rank_words(words)
            
        show_words(sorted_words, predicted_remaining_words, 10)

        guess, colors = get_guess_colors()
        words = get_possible_words(guess, colors, words)
        
        if (words.shape[0] == 1):
            print(f"\n{get_str(words[0])} is the word!")
            break
        elif (words.shape[0] == 0):
            print(f"\nNo words match that description!")
            break

        if (turn == 5):
            print("\nNo more turns!")
            
    print("\nThanks for playing!")


def get_all_valid_words() -> np.ndarray:
    """Retun all valid wordle words."""
    filename = "data/valid-words.txt"
    with open(filename, mode="r") as f:
        lines = f.read().split()

    words = np.array([get_unicode(word) for word in lines])
    return words

def get_unicode(word: str) -> np.ndarray:
    """Vectorize a word by turning it into an array of unicode values."""
    vectorized_word = np.array([ord(char) for char in word], dtype=np.int32)
    return vectorized_word

def get_guess_colors() -> Tuple[np.ndarray, np.ndarray]:
    """Ask the user for the guess and colors."""
    guess = input("\nGuess:  ")
    colors = input("Colors: ")

    guess_vector = get_unicode(guess)
    colors_vector = get_unicode(colors)

    return (guess_vector, colors_vector)


# @profile
def get_possible_words(guess: np.ndarray, colors: np.ndarray, words: np.ndarray) -> np.ndarray:    
    """Filter possible words based on guess and colors."""
    # Green.
    green_letter_mask = (colors == G_UNICODE)
    green_word_mask = np.all(words[:, green_letter_mask] == guess[green_letter_mask], axis=1)

    words = words[green_word_mask]
    
    # Yellow and Black: letter not at ind.
    letter_not_at_index_mask = np.all(words[:, ~green_letter_mask] != guess[~green_letter_mask], axis=1)
    words = words[letter_not_at_index_mask]
    
    # Yellow: guesses must have greater or equal yellow letters at non-green inds.
    yellow_letters = guess[colors == Y_UNICODE]
    unique_yellow_letters =  np.unique(yellow_letters[:, np.newaxis])
    yellow_letter_count_limits = (yellow_letters == unique_yellow_letters[:, np.newaxis]).sum(axis=1)

    possible_words_not_green_letters = words[:, ~green_letter_mask]
    yellow_letter_counts = (
        possible_words_not_green_letters == unique_yellow_letters[:, np.newaxis, np.newaxis]
    ).sum(axis=2).T

    words = words[(yellow_letter_counts >= yellow_letter_count_limits).all(axis=1)]

    # Black: guesses must have less than or equal to yellow black letters.
    black_letters = guess[colors == B_UNICODE]
    unique_black_letters = np.unique(black_letters)

    yellow_black_letter_count = (yellow_letters == unique_black_letters[:, np.newaxis]).sum(axis=1)

    possible_words_not_green_letters = words[:, ~green_letter_mask]
    black_letter_counts = (
        possible_words_not_green_letters == unique_black_letters[:, np.newaxis, np.newaxis]
    ).sum(axis=2).T

    words = words[(black_letter_counts <= yellow_black_letter_count).all(axis=1)]
    
    # Return filtered words.
    return words

# @profile
def rank_words(words: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rank words by predicted number of remaining words."""
    possible_colors = np.array(list(itertools.product([G_UNICODE, Y_UNICODE, B_UNICODE], repeat=5)))

    predicted_remaining_words = np.zeros(words.shape[0], dtype=np.float64)

    print(f"\nRanking {words.shape[0]} words:")
    for (i, guess) in enumerate(words):
        if (i % 25 == 0):
            print(f"Ranking word: {i}")
        
        total_remaining_words = 0
        for colors in possible_colors:
            n_possible_words = get_possible_words(guess, colors, words).shape[0]
            total_remaining_words += (n_possible_words / words.shape[0]) * n_possible_words

        predicted_remaining_words[i] = total_remaining_words

    sorted_inds = np.argsort(predicted_remaining_words)
    
    return (words[sorted_inds], predicted_remaining_words[sorted_inds])


def show_words(words: np.ndarray, predicted_remaining_words: np.ndarray, n_words: int) -> None:
    """Show top words."""
    print("\nRanked Words:")
    for i in range(min(words.shape[0], n_words)):
        print(f"{i}. {get_str(words[i])}   {predicted_remaining_words[i]}")

def get_str(unicode_array: np.ndarray) -> str:
    """Turn a vectorized unicode array into a human-readable string."""
    chars = [chr(val) for val in unicode_array]
    word = "".join(chars)
    return word


if __name__ == "__main__":
    main()
    