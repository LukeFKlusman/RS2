from collections import Counter
from pathlib import Path
import random
import time
import sys


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

GOOD         = "Good"
BAD_POSITION = "Bad_Position"
INCORRECT    = "Incorrect"

TOP_OPENERS = {
    "salet":  10,
    "slate":   9,
    "crane":   9,
    "trace":   8,
    "crate":   7,
    "stare":   6,
    "arise":   5,
    "raise":   5,
    "irate":   4,
    "later":   4,
    "alter":   3,
    "alert":   3,
    "rates":   3,
    "tears":   3,
    "lares":   2,
    "reals":   2,
    "learn":   2,
    "snare":   2,
    "alone":   2,
    "stern":   1,
}

# Easter egg trigger words - this is just for fun - REMOVE IF IT MESS'S WITH LATER LOGIC
EASTER_EGG_WORDS = {"fools", "basic", "dummy", "bland", "plain", "empty", "banal", "sadly"}

EASTER_EGG_INSULTS = [
    "HA! Are you SERIOUS?! {word}?! THAT was your big brain move?!",
    "Oh WOW. {word}. The absolute AUDACITY of choosing {word} like you thought that was clever.",
    "I've seen houseplants make better decisions than choosing {word}. Truly staggering.",
    "Did you actually think {word} was going to give ME trouble? I'm a COMPUTER.",
    "{word}?! My grandmother could guess {word} and she's never played Wordle in her life.",
    "I am genuinely embarrassed FOR you. {word}. Let that sink in. {word}.",
    "Breaking news: local person chooses {word} and is shocked when the robot solves it instantly.",
    "Congratulations! You've chosen {word}, the intellectual equivalent of a participation trophy.",
    "I had {word} locked in before you even finished thinking. That's how painfully obvious it was.",
    "Scientists studying poor decision-making will want to hear about {word}. For all the wrong reasons.",
]

# Terminal colour codes THIS SHIT DOESNT WORK. ABORT MISSION
class C:
    GREEN  = ""
    YELLOW = ""
    RED    = ""
    CYAN   = ""
    BOLD   = ""
    DIM    = ""
    RESET  = ""


# ─────────────────────────────────────────────
#  Dictionary
# ─────────────────────────────────────────────

def load_dictionary(file_path="dictionary.txt"):
    """Load and return sorted list of valid 5-letter words."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {file_path}. Make sure it is in the same folder as main.py"
        )
    words = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().lower()
            if word.isalpha() and len(word) == 5:
                words.add(word)
    return sorted(words)


# ─────────────────────────────────────────────
#  Core Wordle Logic
# ─────────────────────────────────────────────

def score_guess_against_target(guess, target):
    """
    Returns a list of 5 feedback tokens (GOOD, BAD_POSITION, INCORRECT)
    for a given guess scored against the target word.
    """
    result    = [INCORRECT] * 5
    remaining = Counter()

    for i in range(5):
        if guess[i] == target[i]:
            result[i] = GOOD
        else:
            remaining[target[i]] += 1

    for i in range(5):
        if result[i] == GOOD:
            continue
        if remaining[guess[i]] > 0:
            result[i] = BAD_POSITION
            remaining[guess[i]] -= 1

    return result


def word_matches_feedback(candidate, guess, feedback):
    return score_guess_against_target(guess, candidate) == feedback


def filter_candidates(candidates, guess, feedback):
    return [w for w in candidates if word_matches_feedback(w, guess, feedback)]


# ─────────────────────────────────────────────
#  Guess Selection
# ─────────────────────────────────────────────

def choose_opening_guess(words):
    """
    Weighted-random opener. Top word appears ~11% of the time.
    P(same opener twice in a row) across pool = ~1.6% -- well under 50%.
    """
    valid = [(w, wt) for w, wt in TOP_OPENERS.items() if w in words]
    if not valid:
        return random.choice(words)
    openers, weights = zip(*valid)
    return random.choices(openers, weights=weights, k=1)[0]


def choose_best_guess(candidates, top_n=5):
    """
    Scores by letter frequency, picks randomly from top N for variety.
    """
    if not candidates:
        return None
    if len(candidates) <= top_n:
        return random.choice(candidates)
    freq   = Counter(c for word in candidates for c in set(word))
    scored = sorted(candidates, key=lambda w: sum(freq[c] for c in set(w)), reverse=True)
    return random.choice(scored[:top_n])


# ─────────────────────────────────────────────
#  Input Parsing
# ─────────────────────────────────────────────

def parse_feedback(user_input):
    """
    Accepts basically anything:
      GGGGG | G G G G G | g,b,i,g,g | G/B/I/G/G | gBiGg
    Strips all delimiters, reads 5 characters.
    """
    cleaned = (
        user_input.strip()
        .upper()
        .replace(",", "")
        .replace("/", "")
        .replace(".", "")
        .replace("-", "")
        .replace(" ", "")
    )

    if len(cleaned) != 5:
        raise ValueError(
            f"Expected 5 feedback characters, got {len(cleaned)}. "
            "Use G (good), B (bad position), I (incorrect)."
        )

    mapping = {"G": GOOD, "B": BAD_POSITION, "I": INCORRECT}
    parsed  = []

    for ch in cleaned:
        if ch not in mapping:
            raise ValueError(f"Invalid character '{ch}'. Only G, B, and I are allowed.")
        parsed.append(mapping[ch])

    return parsed


# ─────────────────────────────────────────────
#  Display Helpers
# ─────────────────────────────────────────────

def colour_feedback(guess, feedback):
    """Returns a string showing guess letters with feedback symbols."""
    symbol_map = {GOOD: "[G]", BAD_POSITION: "[B]", INCORRECT: "[ ]"}
    return "  " + " ".join(symbol_map[fb] + guess[i].upper() for i, fb in enumerate(feedback))


def print_guess_positions(guess):
    suffixes = ["st", "nd", "rd", "th", "th"]
    for i, ch in enumerate(guess, start=1):
        print(f"  {i}{suffixes[i-1]}: {ch.upper()}")


def print_remaining_info(candidates):
    count = len(candidates)
    if count == 0:
        print("  (No candidates left)")
    elif count <= 10:
        words = ", ".join(w.upper() for w in candidates)
        print(f"  ({count} word(s) left: {words})")
    else:
        print(f"  ({count} possible words still remaining)")


def print_colour_legend():
    print("  [G] = Good (correct position)")
    print("  [B] = Bad position (wrong spot, right letter)")
    print("  [ ] = Incorrect (not in word)\n")


def print_title():
    print("\n  ┌─────────────────────┐")
    print("  │    Wordle Solver    │")
    print("  └─────────────────────┘")


def slow_print(text, delay=0.03):
    """Prints text character by character for dramatic effect."""
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    print()


# ─────────────────────────────────────────────
#  Easter Egg
# ─────────────────────────────────────────────

def trigger_easter_egg(word):
    """Absolutely roasts the player for picking an embarrassingly easy word."""
    print("\n  " + "=" * 54)
    time.sleep(0.4)
    slow_print("\n  *** PATHETIC WORD DETECTED ***", delay=0.05)
    time.sleep(0.5)

    insult = random.choice(EASTER_EGG_INSULTS).format(word=word.upper())
    slow_print(f"\n  {insult}", delay=0.03)
    time.sleep(0.3)

    slow_print(f"\n  I am EMBARRASSED to have even been given this task.", delay=0.03)
    slow_print(f"  A CHILD could guess {word.upper()}. A SLEEPING child.", delay=0.03)
    slow_print(f"  Please. For the love of all things holy. Try harder.", delay=0.03)

    time.sleep(0.4)
    print("\n  " + "=" * 54 + "\n")
    time.sleep(0.6)


def print_bug_report():
    """Shows the hidden easter egg words — dev reference."""
    print("\n  ── Easter Egg Words ────────────────")
    for word in sorted(EASTER_EGG_WORDS):
        print(f"  > {word.upper()}")
    print("  ────────────────────────────────────\n")


# ─────────────────────────────────────────────
#  Session Stats
# ─────────────────────────────────────────────

session_stats = {"games": 0, "total_attempts": 0, "best": None}

def update_stats(attempts):
    session_stats["games"] += 1
    session_stats["total_attempts"] += attempts
    if session_stats["best"] is None or attempts < session_stats["best"]:
        session_stats["best"] = attempts

def print_stats():
    g = session_stats["games"]
    if g == 0:
        print("\n  No games played yet this session.")
        return
    avg  = session_stats["total_attempts"] / g
    best = session_stats["best"]
    print("\n  ── Session Stats ───────────────────")
    print(f"  Games played : {g}")
    print(f"  Avg attempts : {avg:.1f}")
    print(f"  Best game    : {best} attempt(s)")
    print("  ────────────────────────────────────\n")


# ─────────────────────────────────────────────
#  Game Modes
# ─────────────────────────────────────────────

def manual_solver(words):
    """
    Interactive mode -- script guesses, human gives feedback.
    ROS2 note: input() -> subscriber callback, print() -> publisher.
    """
    candidates = words[:]
    attempt    = 1

    print("\n  Think of a 5-letter secret word. DO NOT type it.")
    print("  The script will guess. You provide the feedback.\n")
    print_colour_legend()
    print("  Accepted formats:  GGGGG  |  G G G G G  |  g,b,i,g,g  |  gBiGg  etc.\n")

    while True:
        if not candidates:
            print("\n  No candidates left. Your feedback may have been inconsistent.")
            return

        guess = choose_opening_guess(words) if attempt == 1 else choose_best_guess(candidates)

        print(f"\n  ── Attempt {attempt} ──")
        print(f"  Guess: {guess.upper()}")


        try:
            feedback_input = input("\n  Enter 5 results: ").strip()
            feedback       = parse_feedback(feedback_input)
        except ValueError as e:
            print(f"\n  Error: {e}  -> Try again.")
            continue

        print(colour_feedback(guess, feedback))

        if all(f == GOOD for f in feedback):
            print(f"\n  Solved in {attempt} attempt(s)! The word was: {guess.upper()}")
            update_stats(attempt)
            return

        candidates = filter_candidates(candidates, guess, feedback)
        print_remaining_info(candidates)
        attempt += 1


def auto_test_solver(words):
    """
    Auto mode -- script plays against a known target.
    Triggers easter egg if the word is embarrassingly easy.
    """
    target = input("\n  Enter the hidden 5-letter word for testing: ").strip().lower()

    if not target.isalpha() or len(target) != 5:
        print("  Please enter exactly 5 letters.")
        return

    if target not in words:
        print("  That word is not in dictionary.txt")
        return

    if target in EASTER_EGG_WORDS:
        trigger_easter_egg(target)

    candidates = words[:]
    attempt    = 1

    print(f"\n  Testing solver against: {target.upper()}\n")

    while True:
        if not candidates:
            print("  No candidates left -- something went wrong.")
            return

        guess    = choose_opening_guess(words) if attempt == 1 else choose_best_guess(candidates)
        feedback = score_guess_against_target(guess, target)

        print(f"  Attempt {attempt}: {guess.upper()}  ->  {colour_feedback(guess, feedback).strip()}")

        if guess == target:
            print(f"\n  Solved in {attempt} attempt(s)!")
            update_stats(attempt)
            return

        candidates = filter_candidates(candidates, guess, feedback)
        print_remaining_info(candidates)
        attempt += 1


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────

def main():
    words = load_dictionary("dictionary.txt")

    while True:
        print_title()
        print("  1 = Manual mode  (you think of a word, script guesses)")
        print("  2 = Auto test    (script plays against a known word)")
        print("  3 = Stats        (session performance)")
        print("  4 = Quit")
        print("  5 = Bug\n")

        mode = input("  Choose mode: ").strip()

        if mode == "1":
            manual_solver(words)
        elif mode == "2":
            auto_test_solver(words)
        elif mode == "3":
            print_stats()
        elif mode == "4":
            print_stats()
            print("\n  Goodbye.\n")
            break
        elif mode == "5":
            print_bug_report()
        else:
            print("  Invalid choice. Pick 1, 2, 3, 4, or 5.")


if __name__ == "__main__":
    main()
