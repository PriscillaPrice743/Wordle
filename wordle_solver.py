import pandas as pd
import numpy as np

import itertools


class WordleSolver:
    """A class to represent a wordle solver."""

    def __init__(self):
        """Initialize all valid words."""
        self.words = self.get_all_valid_words()

    def get_all_valid_words(self) -> pd.DataFrame:
        """Get all valid wordle words."""
        filename = "data/valid-words.txt"
        with open(filename, mode="r") as f:
            text = f.read()

        words = pd.DataFrame([list(word) for word in text.split()], dtype="string")
        return words

    def filter_words(self, guess: pd.Series, colors: str, words: pd.DataFrame) -> pd.DataFrame:
        """Filter words by guess and colors."""
        possible_words = words.copy(deep=True)

        guess_colors = list(zip(guess, colors))
        for (index, (letter, color)) in enumerate(guess_colors):
            if (color == "g"):
                letter_at_index = (possible_words[index] == letter)
                possible_words = possible_words[letter_at_index]
            elif (color == "y"):
                letter_not_at_index = (possible_words[index] != letter)

                not_green_inds = [i for i in range(len(colors)) if colors[i] != "g"]
                letter_at_not_green_inds = (possible_words[not_green_inds] == letter).any(axis="columns")
                
                yellow = letter_at_not_green_inds & letter_not_at_index
                possible_words = possible_words[yellow]
            else:
                if (letter, "y") in guess_colors:
                    letter_not_at_index = (possible_words[index] != letter)
                    possible_words = possible_words[letter_not_at_index]
                else:
                    not_green_inds = [i for i in range(len(colors)) if colors[i] != "g"]
                    letter_not_at_not_green_inds = (possible_words[not_green_inds] != letter).all(axis="columns")

                    possible_words = possible_words[letter_not_at_not_green_inds]
        
        return possible_words

    def rank_words(self, words: pd.DataFrame) -> pd.Series:
        """Rank words based on predicted remaining guesses."""
        possible_color_outcomes = ["".join(colors) for colors in itertools.product("gyb", repeat=5)]

        n_words = words.shape[0]
        print(f"\nRanking total of {n_words} words.")

        predicted_remaining_words = pd.Series(np.zeros(shape=n_words), index=words.index)
        for n_word in range(n_words):
            if (n_word % 10 == 0):
                print(f"Ranking word {n_word}")

            i = words.index[n_word]
            guess = words.loc[i]
            predicted_remaining = 0
            for colors in possible_color_outcomes:
                n_remaining_words = self.filter_words(guess, colors, words).shape[0]
                predicted_remaining += (n_remaining_words / n_words) * n_remaining_words   # TODO: Fix information math (bits).

            predicted_remaining_words[i] = predicted_remaining

        predicted_remaining_words.sort_values(ascending=True, inplace=True)
        return predicted_remaining_words

    def show_words(self, guesses: pd.DataFrame, predicted_remaining_words: pd.Series, n_words: int) -> None:
        """Display the best n valid words."""
        print(f"\nBest guesses:")
        for i in range(min(predicted_remaining_words.size, n_words)):
            guess = "".join(guesses.iloc[i])
            predicted_remaining = predicted_remaining_words.iloc[i]
            print(f"{i}. Guess: {guess}    Predicted Remaining: {predicted_remaining}")
    
    def run(self):
        """Run the wordle solver program."""
        print("\n----------WORDLE SOLVER----------")

        while (True):
            guess = input("\nGuess:  ")
            colors = input("Colors: ")
            
            self.words = self.filter_words(guess, colors, self.words)

            predicted_remaining_words = self.rank_words(self.words) # TODO: speed, no hard mode (all guesses)
            self.show_words(self.words, predicted_remaining_words, n_words=20)

            if (self.words.shape[0] == 1):
                word = "".join(self.words.iloc[0])
                print(f"\nThe word is {word}!")
                break


if __name__ == "__main__":
    ws = WordleSolver()
    ws.run()
