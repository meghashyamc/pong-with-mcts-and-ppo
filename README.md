# Code Readme

## Components of the Code

The code is divided into four main components:

1. The `src/algorithms` folder contains implementations of the algorithms PPO and MCTS (and their modified versions PPO-MCTS and MCTS aided by PPO).
2. The `src/pong` folder contains implementations of Simple/Complex Pong games.
3. The `src/train` folder contains a script to train an agent (using PPO/PPO-MCTS) to play Simple/Complex Pong.
4. The `src/play` folder contains a script to play (using MCTS/MCTS aided by PPO) Simple/Complex Pong.

Other than the `src` folder, there are also the `results` and `trained` folders. The `results` folder contains the data generated during training and playing. The `trained` folder contains the trained algorithm files (.pth files) for PPO and PPO-MCTS training sessions.

## Setting up the Environment

To run the code, you need to have Python 3.11 or higher installed on your machine.
Python can be installed [from this link](https://www.python.org/downloads/release/python-3119/).
After that, from the `code` folder, run the following command:

```bash
pip install -r requirements.txt
```

This will install all the required packages for the code to run.

## Playing Simple/Complex Pong (Human Player)

From the `code` folder, run the following command:

```bash
python -m src.pong.main --env_name SimplePong
```

Or, you can run:

```bash
python -m src.pong.main --env_name ComplexPong
```

Left and right keys can be used to move the paddle. In the case of ComplexPong, for the left paddle,
'A' and 'D' keys can be used to move the paddle.

## Making MCTS or MCTS Aided by PPO play Simple/Complex Pong and Generate Data

From the `code` folder, run the following command:

```bash
python -m src.play.play_mcts --env_name SimplePong --iterations 100 --reward_frequency frequent
```

This will make MCTS play SimplePong with 100 iterations and frequent rewards. Logs will be visible on the terminal. To view the game while it is played, you can pass `--render True`. To make pre-trained PPO
aid MCTS, use `--ppo True`. Similarly, `--reward_frequency sparse` will make sure MCTS plays the
game with sparse rewards. `--env_name ComplexPong` will make MCTS (or MCTS aided by PPO) play Complex Pong.

Once 300 episodes have been played, the program will stop running and results will be generated (.csv files) in the `results` folder in the appropriate sub-folder.

At any point, to stop the program, press `Ctrl + C`.

## Train PPO or PPO-MCTS to Play Simple/Complex Pong and Generate Data

From the `code` folder, run the following command:

```bash
python -m src.train.train_ppo --env_name SimplePong --reward_frequency frequent
```

This will start PPO's training to play SimplePong  with frequent rewards. Logs will be visible on the terminal. To view the game while it is played, you can pass `--render True`.  To also view a graph of the moving average of paddle hits episode by episode, pass `--plot True`.  To make PPO train for Complex Pong, use `--env_name ComplexPong`. For sparse rewards, use `--reward_frequency sparse`.
To use PPO-MCTS instead of only PPO, pass the argument `--mcts True`.

At any point, to stop the program, press `Ctrl + C`.

## Compile Generated Data into Results

There are three Python scripts in the `results` folder. The script `results/max_episode_calculator.py` generates a csv file listing the number of episodes needed for each training session (PPO/PPO-MCTS). The script `results/time_taken_calculator.py` generates a csv file listing the total time taken for each training session. The script `results/training_average_calculator.py` generates a csv file that contains the moving average of paddle hits for each episode across all training sessions. More instructions to run these scripts are in the script files. These scripts accept a folder name and a 'phrase' ('frequent' or 'sparse') as arguments. The folder name is the results folder that contains the paddle hits data or timestamp data that needs to be compiled into usable results.

Here's one example of running one of the scripts from the `code` folder:

```bash
python -m results.max_episode_calculator results/SimplePong/ppo-mcts/original frequent
```

This will create a compilation of the total number of episodes needed to train PPO-MCTS for SimplePong with frequent rewards (based on the .csv files in the folder) across 30 training runs. This compilation will be saved in the same folder.
