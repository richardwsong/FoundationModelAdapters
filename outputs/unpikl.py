import pickle

# Replace with your file path
file_path = "final_eval.pkl"

# Open and load the pickle file
with open(file_path, "rb") as f:
    data = pickle.load(f)

# Print the content
print(data)