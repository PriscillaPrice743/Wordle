import numpy as np

import itertools

from typing import Tuple


def get_all_valid_words() -> np.chararray:
    """Retun all valid wordle words."""
    filename = "data/valid-words.txt"
    with open(filename, mode="r") as f:
        lines = f.read().split()

    words = np.char.array([list(word) for word in lines])
    return words

def get_guess_colors() -> Tuple[np.chararray, np.chararray]:
    """Ask the user for the guess and colors."""
    guess = input("\nGuess:  ")
    colors = input("Colors: ")

    guess_vector = np.char.array(list(guess))
    colors_vector = np.char.array(list(colors))

    return (guess_vector, colors_vector)

@profile
def get_possible_words(guess, colors, words) -> np.chararray:    
    """Filter possible words based on guess and colors."""
    # Green.
    green_letter_mask = (colors == "g")
    green_word_mask = np.all(words[:, green_letter_mask] == guess[green_letter_mask], axis=1)

    words = words[green_word_mask]
    
    # Yellow and Black: letter not at ind.
    letter_not_at_index_mask = np.all(words[:, ~green_letter_mask] != guess[~green_letter_mask], axis=1)
    words = words[letter_not_at_index_mask]
    
    # Letter count limits.
    not_green_letters = guess[~green_letter_mask]
    not_green_colors = colors[~green_letter_mask]
    
    unique_not_green_letters, unique_not_green_inds = np.unique(not_green_letters, return_index=True)
    unique_not_green_colors = not_green_colors[unique_not_green_inds]
    
    unique_yellow_letter_mask = unique_not_green_colors == "y"
    unique_yellow_letters = unique_not_green_letters[unique_yellow_letter_mask]
    
    yellow_letter_count_limits = (not_green_letters == unique_yellow_letters[:, np.newaxis]).sum(axis=1)
    
    letter_count_limits = np.zeros(unique_not_green_letters.size, dtype=np.int32)
    letter_count_limits[unique_yellow_letter_mask] = yellow_letter_count_limits
    
    # Letter counts.
    possible_words_not_green_letters = words[:, ~green_letter_mask]
    
    letter_counts = (
        possible_words_not_green_letters == unique_not_green_letters[:, np.newaxis, np.newaxis]
    ).sum(axis=2).T
    
    # Yellow mask.
    yellow_above_count_limits_mask = (
        letter_counts[:, unique_yellow_letter_mask] >= letter_count_limits[unique_yellow_letter_mask]
    ).all(axis=1)
    
    # Black mask.
    black_below_count_limits_mask = (
        letter_counts[:, ~unique_yellow_letter_mask] <= letter_count_limits[~unique_yellow_letter_mask]
    ).all(axis=1)
    
    # Apply Yellow and Black masks.
    words = words[yellow_above_count_limits_mask & black_below_count_limits_mask]
    
    return words

@profile
def rank_words(words) -> Tuple[np.chararray, np.ndarray]:
    """Rank words by predicted number of remaining words."""
    possible_colors = np.char.array(list(itertools.product("gyb", repeat=5)))

    predicted_remaining_words = np.zeros(words.shape[0])

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

def show_words(words, predicted_remaining_words, n_words) -> None:
    """Show top words."""
    print("\nRanked Words:")
    for i in range(min(words.shape[0], n_words)):
        print(f"{i}. {''.join(words[i])}   {predicted_remaining_words[i]}")


if __name__ == "__main__":
    words = get_all_valid_words()

    print("----------WORDLE SOLVER----------")

    cont = True
    while (cont):
        sorted_words, predicted_remaining_words = rank_words(words)
        show_words(sorted_words, predicted_remaining_words, 15)

        guess, colors = get_guess_colors()
        words = get_possible_words(guess, colors, words)
        
        if (words.shape[0] == 1):
            print(f"\n{''.join(words[0])} is the word!")
            cont = False
        elif (words.shape[0] == 0):
            print(f"\nNo words match that description!")
            cont = False
            
    print("\nThanks for playing!")
    