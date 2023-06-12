from wordle_solver import WordleSolver


ws = WordleSolver()

print("\nEnter 'quit' to quit.\n")

while (True):
    guess = input("\nGuess:  ")
    if (guess == "quit"):
        break

    colors = input("Colors: ")
    if (guess == "quit"):
        break
    
    ws.filter_words(guess, colors)
    ws.show_words(10)
