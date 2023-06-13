import pandas as pd
import numpy as np

import itertools


class WordleSolver:
    """A class to represent a wordle solver."""

    def __init__(self):
        """Initialize all valid words."""
        self.words = self.get_all_valid_words()

    def get_all_valid_words(self) -> pd.Series:
        """Get all valid wordle words."""
        filename = "data/valid-words.txt"
        with open(filename, mode="r") as f:
            text = f.read()

        words = pd.Series(text.split(), dtype="string")
        return words

    def filter_words(self, guess: str, colors: str, words: pd.Series) -> pd.Series:
        """Filter words by guess and colors."""
        possible_guesses = words.copy()

        guess_colors = list(zip(guess, colors))
        for (index, (letter, color)) in enumerate(guess_colors):
            if (color == "g"):
                letter_at_index = (possible_guesses.str.get(index) == letter)
                possible_guesses = possible_guesses[letter_at_index]
            elif (color == "y"):
                not_green_inds = [i for i in range(len(colors)) if colors[i] != "g"]
                letter_at_not_green_inds_mask = pd.DataFrame({i : (possible_guesses.str.get(i) == letter) for i in not_green_inds})
                letter_at_not_green_inds = letter_at_not_green_inds_mask.any(axis="columns")

                letter_not_at_index = (possible_guesses.str.get(index) != letter)
                
                yellow = letter_at_not_green_inds & letter_not_at_index
                possible_guesses = possible_guesses[yellow]
            else:
                if (letter, "y") in guess_colors:
                    letter_not_at_index = (possible_guesses.str.get(index) != letter)
                    possible_guesses = possible_guesses[letter_not_at_index]
                else:
                    not_green_inds = [i for i in range(len(colors)) if colors[i] != "g"]
                    letter_not_at_not_green_inds_mask = pd.DataFrame({i : (possible_guesses.str.get(i) != letter) for i in not_green_inds})
                    letter_not_at_not_green_inds = letter_not_at_not_green_inds_mask.all(axis="columns")

                    possible_guesses = possible_guesses[letter_not_at_not_green_inds]
        
        return possible_guesses

    def rank_words(self, words: pd.Series) -> pd.DataFrame:
        """Rank words based on predicted remaining guesses."""
        possible_color_outcomes = ["".join(colors) for colors in itertools.product("gyb", repeat=5)]

        print(f"\nRanking total of {words.size} words.")

        predicted_remaining_words = pd.Series(np.zeros(shape=words.size), index=words.index)
        for n_word in range(words.size):
            if (n_word % 10 == 0):
                print(f"Ranking word {n_word}")

            i = words.index[n_word]
            guess = words[i]
            predicted_remaining = 0
            for colors in possible_color_outcomes:
                n_remaining_words = self.filter_words(guess, colors, words).size
                predicted_remaining += (n_remaining_words / words.size) * n_remaining_words

            predicted_remaining_words[i] = predicted_remaining

        predicted_remaining_words = pd.DataFrame({"guess" : words,
                                                  "predicted_remaining" : predicted_remaining_words})
        return predicted_remaining_words

    def show_words(self, predicted_remaining_words: pd.DataFrame, n_words: int) -> None:
        """Display the best n valid words."""
        predicted_remaining_words.sort_values(by="predicted_remaining", axis="index",
                                              ascending=True, inplace=True)
        print(f"\nBest guesses:")
        for i in range(min(predicted_remaining_words.size, n_words)):
            guess = predicted_remaining_words["guess"].iloc[i]
            predicted_remaining = predicted_remaining_words["predicted_remaining"].iloc[i]
            print(f"{i}. Guess: {guess}    Predicted Remaining: {predicted_remaining}")
    
    def run(self):
        """Run the wordle solver program."""
        print("\n----------WORDLE SOLVER----------")
        print("Enter 'quit' to quit.")

        while (True):
            guess = input("\nGuess:  ")
            if (guess == "quit"):
                break

            colors = input("Colors: ")
            if (guess == "quit"):
                break
            
            self.words = self.filter_words(guess, colors, self.words)

            predicted_remaining_words = self.rank_words(self.words)
            self.show_words(predicted_remaining_words, n_words=20)


if __name__ == "__main__":
    ws = WordleSolver()
    ws.run()
