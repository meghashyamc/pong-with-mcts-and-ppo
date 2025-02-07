"""
Run this file to generate a csv file that has the time taken for training
for all 30 training sessions (PPO/PPO-MCTS). This csv file is generated
based on the csv files that are generated 'during' the training which
contain timestamps for each episode.
"""

import os
import pandas as pd


def calculate_total_time_taken(df):
    """
    Calculates total time taken for a single PPO/PPO-MCTS training run.
    The csv files generated during training store episode timestamps.
    This function uses those timestamps to calculate the total time taken.
    """
    # Initialize total time with the time of the first episode
    total_time_taken = df["Time (seconds)"].iloc[0]

    # Calculate time differences between consecutive episodes
    df["Time Difference"] = df["Time (seconds)"].diff()

    # Handle case where time difference exceeds 1000 seconds. This
    # can happen if the computer went into hibernate mode or faced an issue during
    # training. Such data points are rare but when they're encountered, they
    # are substituted by the previous time difference.
    for i in range(1, len(df)):
        if df.loc[i, "Time Difference"] > 1000:
            print(
                "Found a time difference of more than 1000 seconds!",
                df.loc[i, "Time Difference"],
                "Substituting this with previous time difference of",
                df.loc[i - 1, "Time Difference"],
            )

            # Duplicate the previous episode time if time difference is too large
            df.loc[i, "Time Difference"] = df.loc[i - 1, "Time Difference"]

    # Sum the time differences (excluding the first NaN entry)
    total_time_taken += df["Time Difference"].iloc[1:].sum()

    return total_time_taken


def save_total_time_to_csv(folder_path, phrase):
    """
    Generate a csv file containing the total time taken for each of 30 PPO/PPO-MCTS training runs.
    Files that start with episodes_vs_time_frequent or episodes_vs_time_sparse are considered.
    """
    # Get all CSV files in the folder that start with "episodes_vs_time_<phrase>"
    csv_files = [
        f
        for f in os.listdir(folder_path)
        if f.startswith(f"episodes_vs_time_{phrase}") and f.endswith(".csv")
    ]

    # Check if the number of files is exactly 30
    if len(csv_files) != 30:
        print(
            f"Error: Expected 30 CSV files but found {len(csv_files)} for 'episodes_vs_time_{phrase}'."
        )
        return

    # Create the full file paths
    csv_paths = [os.path.join(folder_path, file) for file in csv_files]

    total_times = []  # List to store total time taken for each run

    # Iterate over all CSV files
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)

        # Ensure the DataFrame has the required columns
        if "Episode" not in df.columns or "Time (seconds)" not in df.columns:
            print(f"Error: Missing required columns in {csv_path}. Skipping this file.")
            continue

        # Calculate total time taken for this run (CSV file)
        total_time_taken = calculate_total_time_taken(df)
        total_times.append(total_time_taken)

    # Calculate the average total time taken across all 30 runs
    average_total_time = sum(total_times) / len(total_times)

    # Create a DataFrame with the results
    df_total_times = pd.DataFrame(
        {"CSV File": csv_files, "Total Time Taken (seconds)": total_times}
    )

    # Save the DataFrame to CSV
    output_file = os.path.join(folder_path, f"final_total_time_taken_{phrase}.csv")
    df_total_times.to_csv(output_file, index=False)

    # Print the average total time
    print(f"Average total time taken across 30 runs: {average_total_time:.6f} seconds")
    print(f"CSV file saved as {output_file}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python script.py <folder_path> <phrase>")
        sys.exit(1)

    folder_path = sys.argv[1]  # Folder where episode timestamp csv files are located
    phrase = sys.argv[2]  # frequent or sparse

    # Run the function
    save_total_time_to_csv(folder_path, phrase)
