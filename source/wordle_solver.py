import pandas as pd


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

    def filter_words(self, guess: str, colors: str) -> None:
        """Filter words by guess and colors."""
        guess_colors = list(zip(guess, colors))
        for (index, (letter, color)) in enumerate(guess_colors):
            if (color == "g"):
                letter_at_index = (self.words.str.get(index) == letter)
                self.words = self.words[letter_at_index]
            elif (color == "y"):
                not_green_inds = [i for i in range(len(colors)) if colors[i] != "g"]
                letter_at_not_green_inds_mask = pd.DataFrame({i : (self.words.str.get(i) == letter) for i in not_green_inds})
                letter_at_not_green_inds = letter_at_not_green_inds_mask.any(axis="columns")

                letter_not_at_index = (self.words.str.get(index) != letter)
                
                yellow = letter_at_not_green_inds & letter_not_at_index
                self.words = self.words[yellow]
            else:
                if (letter, "y") in guess_colors:
                    letter_not_at_index = (self.words.str.get(index) != letter)
                    self.words = self.words[letter_not_at_index]
                else:
                    not_green_inds = [i for i in range(len(colors)) if colors[i] != "g"]
                    letter_not_at_not_green_inds_mask = pd.DataFrame({i : (self.words.str.get(i) != letter) for i in not_green_inds})
                    letter_not_at_not_green_inds = letter_not_at_not_green_inds_mask.all(axis="columns")

                    self.words = self.words[letter_not_at_not_green_inds]

    def show_words(self, n_words: int) -> None:
        """Display the first n valid words."""
        print(f"\n{self.words.size} Valid words:")
        for i in range(min(self.words.size, n_words)):
            print(f"{i}. {self.words.iloc[i]}")
