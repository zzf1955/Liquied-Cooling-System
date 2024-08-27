from stable_baselines3.common.callbacks import BaseCallback
import os
import csv

class LogToCSVCallback(BaseCallback):
    """
    Callback for logging infos (including reward, actions, obs) to a CSV file at the end of each episode.
    """
    def __init__(self, log_dir: str, verbose=1):
        super(LogToCSVCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.csv_file = os.path.join(log_dir, 'training_log.csv')
        self.step_data = []
        self.episode_number = 0
        self.step_number = 0

    def _init_callback(self) -> None:
        # Create folder if needed and initialize CSV file with headers
        os.makedirs(self.log_dir, exist_ok=True)
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Step", "Reward", "Action0", "Action1", "Obs1", "Obs2", "Obs3"])

    def _on_step(self) -> bool:
        self.step_number+=1
        # Extract info from self.locals['infos']
        info = self.locals['infos'][0]  # Assuming info is a list with one dictionary per step

        reward = info['reward']
        actions = info['actions']
        obs1,obs2,obs3 = info['obs']

        # Assuming actions and obs are lists and have fixed lengths
        action0, action1 = actions

        # Store the data in step_data list
        self.step_data.append([self.episode_number, self.num_timesteps, reward, action0, action1, obs1, obs2,obs3])
        if self.locals['dones']:
            self.on_episode_end()
        return True

    def on_episode_end(self) -> None:
        self.episode_number += 1

        # Write the collected step data to the CSV file
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.step_data)

        # Clear the step data list for the next episode
        self.step_data = []
