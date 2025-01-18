# Viterbi Algorithm for the Biological Problem

# Observed sequence
S = "ACCGTGCA"

# States and their indices
states = ["H", "L"]
state_index = {"H": 0, "L": 1}

# Transition probabilities
transition_probs = {
    "H": {"H": 0.5, "L": 0.5},
    "L": {"H": 0.4, "L": 0.6},
}

# Emission probabilities
emission_probs = {
    "H": {"A": 0.2, "C": 0.3, "G": 0.3, "T": 0.2},
    "L": {"A": 0.3, "C": 0.2, "G": 0.2, "T": 0.3},
}

# Initialize Viterbi variables
n = len(S)+1  # Length of the sequence
num_states = len(states)
viterbi = [[0] * n for _ in range(num_states)]  # Viterbi table
backpointer = [[0] * n for _ in range(num_states)]  # Backpointer table

viterbi[0][0] = 1

# Recursion (t > 1)
for t in range(0, n-1):
    for curr_state in states:
        max_prob = 0
        max_prev_state = 0

        for prev_state in states:
            prob = (
                viterbi[state_index[prev_state]][t]
                * transition_probs[prev_state][curr_state]
                * emission_probs[curr_state][S[t]]
            )
            if prob > max_prob:
                max_prob = prob
                max_prev_state = state_index[prev_state]
        print(f"{t}: Max prob: {max_prob}, Max State: {max_prev_state}")

        viterbi[state_index[curr_state]][t + 1] = max_prob
        backpointer[state_index[curr_state]][t + 1] = max_prev_state

# Termination: Find the most probable final state
final_probs = [viterbi[i][n - 1] for i in range(num_states)]
final_state = final_probs.index(max(final_probs))

# Backtrack to find the best state sequence
best_path = [0] * n
best_path[n - 1] = final_state

for t in range(n - 2, -1, -1):
    best_path[t] = backpointer[best_path[t + 1]][t + 1]

# Convert state indices to state names
best_path_states = [states[state] for state in best_path]

# Print results
print("Most likely state sequence:", " -> ".join(best_path_states))
print("Probability of the observed sequence:", max(final_probs))
