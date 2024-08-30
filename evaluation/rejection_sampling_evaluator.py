import pandas as pd
import matplotlib.pyplot as plt
import random
import json
import torch
from sft_evaluator import SFTEvalDataCreator
from utils import clean_text, create_model_and_tokenizer, read_rm_and_tokenizer
from plotnine import ggplot, aes, geom_density, facet_wrap, theme_minimal, labs, theme, element_text, element_line
from evaluation.utils import process_tagging_data, answer


class RejectionSamplingHumanEvalDataCreator(SFTEvalDataCreator):
    """
    A class for creating and managing rejection sampling evaluation datasets.
    """
    def __init__(self, test_df_path, base_model, base_tokenizer, rs_model, rs_tokenizer, device, new_model_name, eval_mode):
        """
        Initializes the data creator with the specified models, tokenizers, and evaluation mode.
        
        Parameters:
            test_df_path (str): Path to the test data CSV file.
            base_model: The base model for generating answers.
            base_tokenizer: The tokenizer associated with the base model.
            rs_model: The rejection sampling model.
            rs_tokenizer: The tokenizer associated with the rejection sampling model.
            device (str): The device to run the models on (e.g., 'cpu' or 'cuda').
            new_model_name (str): Name for the new model being created.
            eval_mode (str): The evaluation mode to use.
        """
        super().__init__(test_df_path, base_model, base_tokenizer, rs_model, rs_tokenizer, device, new_model_name)
        self.eval_mode = eval_mode

    def CreateDB_rs_vs_base(self, save_path):
        """
        Creates a dataset for rejection sampling vs. base model and saves it to the specified path.
        
        Parameters:
            save_path (str): Path where the dataset will be saved.
        """
        self.CreateDB(save_path)

    def CreateDB_rs_vs_gpt(self, rs_questions_path, gpt_answers_path, save_path):
        """
        Creates a dataset for rejection sampling vs. GPT model and saves it to the specified path.
        
        Parameters:
            rs_questions_path (str): Path to the CSV file containing rejection sampling questions.
            gpt_answers_path (str): Path to the JSONL file containing GPT model answers.
            save_path (str): Path where the dataset will be saved.
        """
        rs_df = pd.read_csv(rs_questions_path, encoding='utf-8')
        rs_model, rs_tokenizer = self.sft_model, self.sft_tokenizer
        rs_df['option1'] = rs_df['Question'].apply(lambda x: answer(rs_model, rs_tokenizer, x, self.device))
        rs_df['Source1'] = 'rs_model'
        rs_df['option1'] = rs_df['option1'].apply(lambda x: clean_text(x))

        # Load GPT model responses from JSONL file
        content_list = []
        with open(gpt_answers_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                content = data['response']['body']['choices'][0]['message']['content']
                content_list.append(content)

        gpt_df = pd.DataFrame({
            'option2': content_list,
            'Source2': ['gpt_model' for _ in range(len(content_list))]
        })

        # Merge RS and GPT responses
        merged_df = pd.merge(rs_df, gpt_df, left_index=True, right_index=True)
        merged_df = merged_df[['Question', 'option1', 'option2', 'Source1', 'Source2']]
        shuffled_df = merged_df.apply(self._shuffle_options, axis=1)

        self._save_df(shuffled_df, save_path)

    def _shuffle_options(self, row):
        """
        Randomly shuffles options and their sources.

        Parameters:
            row (pd.Series): A row from the DataFrame.

        Returns:
            pd.Series: The shuffled row.
        """
        if random.random() > 0.5:
            row['option1'], row['option2'] = row['option2'], row['option1']
            row['Source1'], row['Source2'] = row['Source2'], row['Source1']
        return row
    
    def _save_df(self, df, save_path):
        """
        Saves the DataFrame to a CSV file.

        Parameters:
            df (pd.DataFrame): The DataFrame to save.
            save_path (str): Path where the DataFrame will be saved.
        """
        df.to_csv(save_path, index=False, encoding='utf-8')

class RejectionSamplingRewardEvalDataCreator:
    """
    A class for creating datasets and evaluating rewards for rejection sampling models.
    """
    def __init__(self, test_df_path, base_model_name, sft_model_name, rm_model_name, device):
        """
        Initializes the reward evaluation data creator with the specified models and tokenizers.
        
        Parameters:
            test_df_path (str): Path to the test data CSV file.
            base_model_name (str): Name of the base model.
            sft_model_name (str): Name of the SFT model.
            rm_model_name (str): Name of the reward model.
            device (str): The device to run the models on.
        """
        self.test_df = pd.read_csv(test_df_path)[:100]
        self.base_model, self.base_tokenizer = create_model_and_tokenizer(base_model_name)
        self.sft_model, self.sft_tokenizer = create_model_and_tokenizer(sft_model_name)
        self.rm_model, self.rm_tokenizer = read_rm_and_tokenizer(base_model_name, rm_model_name)
        self.device = device

    def save_question_and_answers(self, save_path):
        """
        Saves questions and answers from different models to a CSV file.
        
        Parameters:
            save_path (str): Path where the results will be saved.
        """
        # Swap option1 with option2 and Source1 with Source2 where Source1 is gpt_model
        mask = self.test_df['Source1'] == 'gpt_model'
        self.test_df.loc[mask, ['option1', 'option2']] = self.test_df.loc[mask, ['option2', 'option1']].values
        self.test_df.loc[mask, ['Source1', 'Source2']] = self.test_df.loc[mask, ['Source2', 'Source1']].values

        # Rename columns
        self.test_df.rename(columns={'option1': 'rlhf_answer', 'option2': 'gpt_answer'}, inplace=True)

        # List to hold the results
        results = []

        # Iterate over all rows in the test DataFrame
        for i in range(len(self.test_df)):
            question = self.test_df.iloc[i]['Question']
            llama_model_answer = clean_text(answer(self.base_model, self.base_tokenizer, question, self.device)[1:])
            sft_model_answer = clean_text(answer(self.sft_model, self.sft_tokenizer, question, self.device))

            # Append the results
            results.append({
                'Question': question,
                'llama_answer': llama_model_answer,
                'sft_answer': sft_model_answer,
                'rlhf_answer': self.test_df.iloc[i]['rlhf_answer'],
                'gpt_answer': self.test_df.iloc[i]['gpt_answer']
            })

            if i % 5 == 0:
                print(i)
                results_df = pd.DataFrame(results)
                results_df.to_csv(f'{path}.csv', index=False)

        # Convert the results list to a DataFrame
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'{path}.csv', index=False)

    def get_reward(self, question, answer):
        """
        Calculates the reward for a specific answer to a question.

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

    def evaluate_rewards(self, load_path, save_path):
        """
        Evaluates rewards for answers in the dataset and saves the results to a CSV file.
        
        Parameters:
            load_path (str): Path to the CSV file with questions and answers.
            save_path (str): Path where the rewards data will be saved.
        """
        models_df = pd.read_csv(f'{load_path}.csv')

        rewards_dict = {'Question': models_df['Question'].tolist()}

        for col in ['llama_answer', 'sft_answer', 'rlhf_answer', 'gpt_answer']:
            rewards_dict[col] = [self.get_reward(question, answer) for question, answer in zip(models_df['Question'], models_df[col])]

        rewards_df = pd.DataFrame(rewards_dict)

        # Rename columns
        rewards_df.rename(columns={
            'llama_answer': 'Llama 2',
            'sft_answer': 'SFT',
            'rlhf_answer': 'RLHF',
            'gpt_answer': 'GPT'
        }, inplace=True)

        rewards_df.to_csv(save_path, index=False)

class RejectionSamplingHumanEvaluator:
    """
    A class for evaluating and plotting rejection sampling data with respect to different models.
    """
    def __init__(self, rs_vs_llama_path, rs_vs_gpt_path, sft_vs_llama_path):
        """
        Initializes the evaluator with data from the specified CSV files.
        
        Parameters:
            rs_vs_llama_path (str): Path to the CSV file for rejection sampling vs Llama-2.
            rs_vs_gpt_path (str): Path to the CSV file for rejection sampling vs GPT.
            sft_vs_llama_path (str): Path to the CSV file for SFT vs Llama-2.
        """
        self.rs_vs_llama_df = pd.read_csv(rs_vs_llama_path)
        self.rs_vs_gpt_df = pd.read_csv(rs_vs_gpt_path)
        self.sft_vs_llama_df = pd.read_csv(sft_vs_llama_path)
    
    def evaluate(self, rs_paths, gpt_paths, sft_paths):
        """
        Evaluates the results and plots comparisons between different models.
        
        Parameters:
            rs_paths (str): Path to rejection sampling data.
            gpt_paths (str): Path to GPT model data.
            sft_paths (str): Path to SFT model data.
        """
        combined_tagging_llama = process_tagging_data(rs_paths, self.rs_vs_llama_df)
        combined_tagging_gpt = process_tagging_data(gpt_paths, self.rs_vs_gpt_df)
        combined_tagging_sft = process_tagging_data(sft_paths, self.sft_vs_llama_df)
        
        # Replace mapping values
        llama_mapping = {'rs_model': 'Win', 'Llama_model': 'Loss'}
        gpt_mapping = {'rs_model': 'Win', 'gpt_model': 'Loss'}
        sft_mapping = {'sft_model': 'Win', 'Llama_model': 'Loss'}
        
        pref_mapping = {'Significantly Better': 'Considerably Better'}

        combined_tagging_llama['winner_model'] = combined_tagging_llama['winner_model'].replace(llama_mapping)
        combined_tagging_gpt['winner_model'] = combined_tagging_gpt['winner_model'].replace(gpt_mapping)
        combined_tagging_sft['winner_model'] = combined_tagging_sft['winner_model'].replace(sft_mapping)

        combined_tagging_llama['Preference Strength'] = combined_tagging_llama['Preference Strength'].replace(pref_mapping)
        combined_tagging_gpt['Preference Strength'] = combined_tagging_gpt['Preference Strength'].replace(pref_mapping)
        combined_tagging_sft['Preference Strength'] = combined_tagging_sft['Preference Strength'].replace(pref_mapping)
        
        # Calculate percentages
        llama_winner_perc = self.calculate_percentages(combined_tagging_llama, 'winner_model')
        gpt_winner_perc = self.calculate_percentages(combined_tagging_gpt, 'winner_model')
        sft_winner_perc = self.calculate_percentages(combined_tagging_sft, 'winner_model')

        llama_pref_perc = self.calculate_percentages(combined_tagging_llama[combined_tagging_llama['winner_model'] == 'Win'], 'Preference Strength')
        gpt_pref_perc = self.calculate_percentages(combined_tagging_gpt[combined_tagging_gpt['winner_model'] == 'Win'], 'Preference Strength')
        sft_pref_perc = self.calculate_percentages(combined_tagging_sft[combined_tagging_sft['winner_model'] == 'Win'], 'Preference Strength')
        
        # Combine and reorder data for plotting
        winner_perc_df = pd.DataFrame({
            'Llama-2-7b-chat-RLHF\nvs\nGPT-3.5-turbo': gpt_winner_perc,
            'Llama-2-7b-chat-RLHF\nvs\nLlama-2-7b-chat': llama_winner_perc,
            'Llama-2-7b-chat-SFT\nvs\nLlama-2-7b-chat': sft_winner_perc
        }).fillna(0).T
        
        preference_order = ['Considerably Better',  'Better', 'Slightly Better', 'Unsure']
        pref_perc_df = pd.DataFrame({
            'Llama-2-RLHF\nvs GPT-3.5-turbo': gpt_pref_perc,
            'Llama-2-RLHF\nvs Llama-2': llama_pref_perc,
            'Llama-2-SFT\nvs Llama-2': sft_pref_perc
        }).fillna(0).T

        pref_perc_df = pref_perc_df[preference_order]
        
        # Plot results
        self.plot_results(winner_perc_df, pref_perc_df)
    
    def calculate_percentages(self, df, column):
        """
        Calculates the percentage distribution of values in a given column.
        
        Parameters:
            df (pd.DataFrame): The DataFrame containing the data.
            column (str): The column to calculate percentages for.
        
        Returns:
            pd.Series: A series with value counts as percentages.
        """
        counts = df[column].value_counts(normalize=True) * 100
        return counts
    
    def plot_results(self, winner_perc_df, pref_perc_df):
        """
        Plots the results of the evaluation, including win rates and preference ratings.
        
        Parameters:
            winner_perc_df (pd.DataFrame): DataFrame containing win rates.
            pref_perc_df (pd.DataFrame): DataFrame containing preference ratings.
        """
        fig, axes = plt.subplots(ncols=2, figsize=(15, 3), sharey=True)
        
        color_mapping_winner = {'Win': '#4CAF50', 'Tie': '#FFC107', 'Loss': '#F44336'}
        color_mapping_preference = {'Unsure': '#9E9E9E', 'Slightly Better': '#FFC107', 'Better': '#FF9800', 'Considerably Better': '#FF5722'}
        
        winner_perc_df.plot(kind='barh', stacked=True, ax=axes[0], color=[color_mapping_winner.get(x, '#000000') for x in winner_perc_df.columns], width=0.6)
        axes[0].set_xlabel('Win Rate (%)')
        axes[0].set_xticks(range(0, 101, 10))
        
        pref_perc_df.plot(kind='barh', stacked=True, ax=axes[1], color=[color_mapping_preference.get(x, '#000000') for x in pref_perc_df.columns], width=0.6)
        axes[1].set_xlabel('Preference Rate (%)')
        axes[1].set_xticks(range(0, 101, 10))
        
        for bar in axes[0].patches:
            width = bar.get_width()
            if width > 0:
                axes[0].text(bar.get_x() + width / 2, bar.get_y() + bar.get_height() / 2,
                             f'{width:.1f}%', ha='center', va='center', color='white', fontsize=8, fontweight='bold')
        
        for bar in axes[1].patches:
            width = bar.get_width()
            if width > 0:
                axes[1].text(bar.get_x() + width / 2, bar.get_y() + bar.get_height() / 2,
                             f'{width:.1f}%', ha='center', va='center', color='white', fontsize=8, fontweight='bold')
        
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4, frameon=False, title='')
        
        fig.suptitle('Win Rate and Preference Ratings for Fine-Tuned Models', fontsize=14, fontweight='bold')
        plt.subplots_adjust(top=0.45)
        fig.text(0.01, 0.02, '* Llama-2 model is Llama-2-7b-chat-hf', ha='left', fontsize=10)
        plt.tight_layout()
        plt.show()


class RejectionSamplingRewardEvaluator:
    """
    A class for evaluating and plotting rewards for rejection sampling models.
    
    """
    def __init__(self, rewards_df_path):
        """
        Initializes the reward evaluator with the rewards DataFrame.
        
        Parameters:
            rewards_df_path (str): Path to the CSV file containing rewards data.
        """
        self.rewards_df = pd.read_csv(rewards_df_path)

    def scale_rewards(self):
        """
        Scales the rewards to a range between 0 and 1.
        """
        self.rewards_df = self.rewards_df.drop('Question', axis=1)
        min_value = self.rewards_df.min().min()
        max_value = self.rewards_df.max().max()
        for model in ['Llama 2', 'SFT', 'RLHF', 'GPT']:
            self.rewards_df[model] = (self.rewards_df[model] - min_value) / (max_value - min_value)

    def plot_rewards(self):
        """
        Plots the distribution of rewards for different models as a ridge plot.
        """
        # Scale rewards before plotting
        self.scale_rewards()

        # Prepare data for ridge plot
        rewards_melted = self.rewards_df.melt(id_vars=['Question'], var_name='Model', value_name='Reward')

        # Define the desired order of models
        model_order = ['Llama 2', 'SFT', 'RLHF', 'GPT']

        # Reorder the 'Model' column based on the desired order
        rewards_melted['Model'] = pd.Categorical(rewards_melted['Model'], categories=model_order, ordered=True)

        # Define a nice color palette
        color_palette = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']

        # Create the ridge plot using plotnine (ggplot)
        ridge_plot = (ggplot(rewards_melted, aes(x='Reward', fill='Model'))
                      + geom_density(size=0.75)
                      + facet_wrap('~Model', ncol=1)
                      + theme_minimal()
                      + labs(title="Reward Model Score Distribution For Different Models", x='Reward Model Score', y='')
                      + theme(legend_position='none',
                              subplots_adjust={'hspace': 0.1},
                              strip_text_y=element_text(angle=0),
                              plot_title=element_text(size=11, face='bold', ha='center'),
                              axis_text_y=element_text(size=0),
                              axis_text_x=element_text(size=10),
                              axis_title_x=element_text(size=10),
                              figure_size=(5, 4),
                              panel_grid_major=element_line(color="grey", size=0.5),
                              panel_background=element_line(color='white'),
                              plot_background=element_line(color='white'))
                      )

        # Save the plot as a PNG file using the save method of ggplot object
        ridge_plot.save(filename='ridge_plot.png', dpi=300, width=5, height=4)

        # Print the plot to display it in the notebook
        print(ridge_plot)