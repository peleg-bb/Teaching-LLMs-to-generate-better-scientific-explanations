import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

class RewardEvaluator:
    """
    A class for evaluating and visualizing the performance of a reward model based on the rewards assigned to options in a test dataset.
    """
    def __init__(self, test_df_path, rm_model, rm_tokenizer, device):
        """
        Initializes the RewardEvaluator with test data, a reward model, tokenizer, and device.

        Parameters:
            test_df_path (str): Path to the CSV file with test data.
            rm_model: The reward model for evaluating options.
            rm_tokenizer: The tokenizer for the reward model.
            device (str): The device to use for model inference.
        """
        self.test_df = pd.read_csv(test_df_path)
        self.rm_model = rm_model
        self.rm_tokenizer = rm_tokenizer
        self.device = device
    
    def evaluate(self):
        """
        Evaluates the rewards for options in the test data, calculates accuracy, and plots the rewards.
        """
        self.test_df = self._compute_rewards(self.test_df)
        accuracy = self._calculate_accuracy()
        print(f'Accuracy rate: {accuracy * 100:.2f}%')
        self._plot_rewards()
    
    def _compute_rewards(self, df):
        """
        Computes the rewards for the options in the DataFrame and determines the reward model's winner.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the test data.

        Returns:
            pd.DataFrame: The DataFrame with additional columns for rewards and the reward model's winner.
        """
        df = df.copy()
        for i in range(len(df)):
            option1_reward, option2_reward = self._get_rewards_for_options(df.iloc[i])
            df.loc[i, 'option1_reward'] = option1_reward
            df.loc[i, 'option2_reward'] = option2_reward
            df['rm_winner'] = df.apply(lambda row: row['Source1'] if row['option1_reward'] >= row['option2_reward'] else row['Source2'], axis=1)
        return df

    def _get_rewards_for_options(self, row):
        """
        Retrieves the rewards for the options in a given row.

        Parameters:
            row (pd.Series): The row containing the question and options.

        Returns:
            tuple: The rewards for option1 and option2.
        """
        option1_reward = self._get_reward(row['Question'], row['option1'])
        option2_reward = self._get_reward(row['Question'], row['option2'])
        return option1_reward, option2_reward
    
    def _get_reward(self, question, answer):
        """
        Calculates the reward for a given answer to a question using the reward model.

        Parameters:
            question (str): The question text.
            answer (str): The answer text.

        Returns:
            float: The computed reward for the given answer.
        """
        text = f"[INST] {question} [/INST] {answer} </s>"
        inputs = self.rm_tokenizer.encode_plus(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.rm_model(**inputs)
        return outputs[0].item()
    
    def _calculate_accuracy(self):
        """
        Calculates the accuracy of the reward model based on the number of correct predictions.

        Returns:
            float: The accuracy rate of the reward model.
        """
        condition = (self.test_df['rm_winner'] == self.test_df['final_winner']) | (self.test_df['final_winner'] == 'Tie')
        correct_predictions = self.test_df[condition].shape[0]
        total_rows = self.test_df.shape[0]
        return correct_predictions / total_rows
    
    def _plot_rewards(self):
        """
        Plots the distribution of rewards for the SFT model and Llama model using a ridge plot.
        """
        # Separate the rewards based on the model
        sft_model_rewards = []
        llama_model_rewards = []

        for _, row in self.test_df.iterrows():
            if row['Source1'] == 'sft_model':
                sft_model_rewards.append(row['option1_reward'])
            if row['Source2'] == 'sft_model':
                sft_model_rewards.append(row['option2_reward'])
            if row['Source1'] == 'Llama_model':
                llama_model_rewards.append(row['option1_reward'])
            if row['Source2'] == 'Llama_model':
                llama_model_rewards.append(row['option2_reward'])

        ridge_data = pd.DataFrame({
            'base_reward': llama_model_rewards,
            'sft_reward': sft_model_rewards
        })

        # Ridge Plot
        plt.figure(figsize=(6, 4))

        # Plot KDE plots with filled curves
        sns.kdeplot(data=ridge_data['sft_reward'], fill=True, label='SFT Model')
        sns.kdeplot(data=ridge_data['base_reward'], fill=True, label='Llama Model')

        # Set plot labels and title
        plt.title('Ridge Plot of SFT Model Rewards vs Llama Model')
        plt.xlabel('Reward')
        plt.ylabel('Density')

        # Display legend on the left side
        plt.legend(loc='upper left')

        plt.show()
