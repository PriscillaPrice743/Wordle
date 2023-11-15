import numpy as np
import itertools
import multiprocessing as mp
import time

from typing import Tuple


# Constants.
G_UNICODE = ord("g")
Y_UNICODE = ord("y")
B_UNICODE = ord("b")

POSSIBLE_COLORS = np.array(list(itertools.product([G_UNICODE, Y_UNICODE, B_UNICODE], repeat=5)))


def get_all_valid_words() -> np.ndarray:
    """Retun all valid wordle words."""
    filename = "data/valid-words.txt"
    with open(filename, mode="r") as f:
        lines = f.read().split()

    words = np.array([get_unicode(word) for word in lines])
    return words

def get_guess_colors() -> Tuple[np.ndarray, np.ndarray]:
    """Ask the user for the guess and colors."""
    guess = input("\nGuess:  ")
    colors = input("Colors: ")

    guess_vector = get_unicode(guess)
    colors_vector = get_unicode(colors)

    return (guess_vector, colors_vector)

def show_words(words: np.ndarray, predicted_remaining_words: np.ndarray, n_words: int) -> None:
    """Show top words."""
    print("\nRanked Words:")
    for i in range(min(words.shape[0], n_words)):
        print(f"{i}. {get_str(words[i])}   {predicted_remaining_words[i]}")


def get_unicode(word: str) -> np.ndarray:
    """Vectorize a word by turning it into an array of unicode values."""
    vectorized_word = np.array([ord(char) for char in word], dtype=np.int32)
    return vectorized_word

def get_str(unicode_array: np.ndarray) -> str:
    """Turn a vectorized unicode array into a human-readable string."""
    chars = [chr(val) for val in unicode_array]
    word = "".join(chars)
    return word


def rank_words(words: np.ndarray, pool: mp.Pool) -> Tuple[np.ndarray, np.ndarray]:
    """Rank words by predicted number of remaining words."""
    # Use async multiprocessing to find guess entropy.
    async_pred_remain_words = [
        pool.apply_async(calc_guess_entropy, args=(guess, words, i))
        for (i, guess) in enumerate(words)
    ]
    pred_remain_words = np.array([
        i.get() for i in async_pred_remain_words
    ], dtype=np.float64)

    sorted_inds = np.argsort(pred_remain_words)
    return (words[sorted_inds], pred_remain_words[sorted_inds])

def calc_guess_entropy(guess: np.ndarray, words: np.ndarray, i: int) -> float:
    """Calculate the information entropy (expected # of remaining words) for a guess."""
    if i % 25 == 0:
        print(f"Ranking word: {i}")

    pred_remain_words = 0
    for colors in POSSIBLE_COLORS:
        n_poss_words = get_poss_words(guess, colors, words).shape[0]
        pred_remain_words += (n_poss_words / words.shape[0]) * n_poss_words
    
    return pred_remain_words

# @profile
def get_poss_words(guess: np.ndarray, colors: np.ndarray, words: np.ndarray) -> np.ndarray:    
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


if __name__ == "__main__":
    # Create multiprocessing pool.
    pool = mp.Pool()
    
    words = get_all_valid_words()[:1_000]

    t1 = time.time()
    sorted_words, pred_remain_words = rank_words(words, pool)
    t2 = time.time()

    print(f"Exec time: {t2 - t1} sec.")

    # Close multiprocessing pool.
    pool.close()
    