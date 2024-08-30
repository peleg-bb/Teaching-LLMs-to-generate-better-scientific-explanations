from abc import ABC, abstractmethod
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset
from utils import clean_text, read_model_and_tokenizer, read_rm_and_tokenizer

class BaseProcessor(ABC):
    """
    An abstract base class for data processors, defining the structure for processing and formatting data.

    Subclasses must implement the following abstract methods:
    - `formatting_prompts_func`: Formats the prompts for the model input.
    - `load_and_process_data`: Loads and preprocesses data from specified paths.
    - `convert_to_hf_datasets`: Converts processed data to Huggingface `Dataset` objects.
    """
    @abstractmethod
    def formatting_prompts_func(self, examples):
        pass
    
    @abstractmethod
    def load_and_process_data(self, path1, path2=None):
        pass

    @abstractmethod
    def convert_to_hf_datasets(self, df):
        pass

class SFTProcessor(BaseProcessor):
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def create_words_lists(self, df):
        """
        Converts the text in the dataframe columns into lists of word tokens.

        Args:
            df (pd.DataFrame): The dataframe containing the columns 'Question', 'Answer', and 'text'.

        Returns:
            pd.DataFrame: The dataframe with new columns 'Question_list', 'Answer_list', and 'text_list'
                          containing tokenized word lists.
        """
        df['Question_list'] = df['Question'].apply(self.text_to_words_list)
        df['Answer_list'] = df['Answer'].apply(self.text_to_words_list)
        df['text_list'] = df['text'].apply(self.text_to_words_list)
        return df
    
    def filter_and_clean_data(self, df):
        """
        Filters and cleans the dataframe by removing specific unwanted entries and sorting by text length.

        Args:
            df (pd.DataFrame): The dataframe to be filtered and cleaned.

        Returns:
            pd.DataFrame: The cleaned dataframe.
        """
        df = df[df['Answer'] != "No answer available"]
        df = df[~((df['Dataset'] == 'Wikipedia') & (df['Question'].str.contains('list', case=False, na=False)))]
        df['text_length'] = df['text_list'].apply(len)
        df = df[df['text_length'] >= 70].sort_values(by='text_length').reset_index(drop=True)
        df['cumulative_count'] = range(1, len(df) + 1)
        return df

    def text_to_words_list(self, text):
        """
        Converts a text string into a list of token IDs.

        Args:
            text (str): The text string to be tokenized.

        Returns:
            list: A list of token IDs.
        """
        return self.tokenizer.encode(text)

    def formatting_prompts_func(self, examples):
        """
        Formats the prompts for supervised fine-tuning, combining questions and answers.

        Args:
            examples (dict): A dictionary containing 'Question' and 'Answer' keys.

        Returns:
            list: A list of formatted prompt strings.
        """
        return [f"[INST] {q} [/INST] {a} </s>" for q, a in zip(examples['Question'], examples['Answer'])]

    def load_and_process_data(self, gpt_answers_path, dataset_path):
        """
        Loads and processes data from CSV files for supervised fine-tuning.

        Args:
            gpt_answers_path (str): The file path for the GPT-generated answers CSV.
            dataset_path (str): The file path for the dataset CSV.

        Returns:
            pd.DataFrame: The processed dataframe ready for fine-tuning.
        """
        # Add Dataset column to the df
        df = pd.read_csv(gpt_answers_path)
        dataset_df = pd.read_csv(dataset_path)
        df = pd.merge(df, dataset_df[['Question', 'Dataset']], on='Question', how='left')
        df = df[:100]

        df['text'] = self.formatting_prompts_func(df)
        df = self.create_words_lists(df)
        df = self.filter_and_clean_data(df)
        
        return df

    def convert_to_hf_datasets(self, df):
        """
        Converts the processed dataframe into Huggingface `Dataset` objects for training and validation.

        Args:
            df (pd.DataFrame): The processed dataframe.

        Returns:
            tuple: A tuple containing:
                - train: Huggingface `Dataset` object for training.
                - val: Huggingface `Dataset` object for validation.
                - train_df: Training dataframe.
                - val_df: Validation dataframe.
                - test_df: Test dataframe.
        """
        train_df, val_df = train_test_split(df, test_size=0.02, random_state=42)
        val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)

        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        train = Dataset.from_pandas(train_df)
        val = Dataset.from_pandas(val_df)
        train = train.select_columns(['text'])
        val = val.select_columns(['text'])
        return train, val, train_df, val_df, test_df


class RewardModelProcessor(BaseProcessor):
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def formatting_prompts_func(self, examples):
        """
        Formats the prompts for reward model training, including chosen and rejected responses.

        Args:
            examples (dict): A dictionary containing 'Question', 'chosen', 'rejected', and 'margin_score' keys.

        Returns:
            dict: A dictionary with tokenized inputs for chosen and rejected responses, along with attention masks and margin scores.
        """
        kwargs = {"padding": "max_length",
                  "truncation": True,
                  "max_length": 256,
                  "return_tensors": "pt"
                  }

        question = examples["Question"]
        chosen = examples["chosen"]
        rejected = examples["rejected"]
        prompt_plus_chosen_response = f"[INST] {question} [/INST] {chosen} </s>"
        prompt_plus_rejected_response = f"[INST] {question} [/INST] {rejected} </s>"

        tokens_chosen = self.tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = self.tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)

        return {
            "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
            "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
            "margin": torch.full((1,1), examples["margin_score"])
        }

    def load_and_process_data(self, rm_dataset_path, path2=None):
        """
        Loads and processes data from a CSV file for reward model training.

        Args:
            rm_dataset_path (str): The file path for the reward model dataset CSV.
            path2 (str, optional): An additional file path if needed.

        Returns:
            pd.DataFrame: The processed dataframe with relevant columns.
        """
        df = pd.read_csv(rm_dataset_path)
        preference_mapping = {
            'Significantly better': 1,
            'Better': 2/3,
            'Slightly better': 1/3,
            'Unsure': 0
        }
        df['margin_score'] = df['GPT Preference'].map(preference_mapping)
        df = df[['Question', 'chosen', 'rejected', 'margin_score']]
        df = df[:100]
        return df

    def convert_to_hf_datasets(self, df):
        """
        Converts the processed dataframe into Huggingface `Dataset` objects for reward model training.

        Args:
            df (pd.DataFrame): The processed dataframe.

        Returns:
            tuple: A tuple containing:
                - train: Huggingface `Dataset` object for training.
                - val: Huggingface `Dataset` object for validation.
        """
        rm_dataset = Dataset.from_pandas(df)
        hf_dataset = rm_dataset.map(self.formatting_prompts_func)
        hf_dataset = hf_dataset.train_test_split(0.1)
        train = hf_dataset['train']
        val = hf_dataset['test']
        return train, val


class RejectionSamplingProcessor(BaseProcessor):
    def __init__(self, sft_model_name, rm_model_name, new_model_name, num_samples, device='cpu'):
        self.sft_model, self.sft_tokenizer = read_model_and_tokenizer(sft_model_name)
        self.rm_model, self.rm_tokenizer = read_rm_and_tokenizer(rm_model_name, new_model_name)
        self.num_samples = num_samples
        self.device = device

    def rm_score(self, prompt, response):
        """
        Computes the reward model score for a given prompt and response.

        Args:
            prompt (str): The prompt text.
            response (str): The response text.

        Returns:
            float: The score assigned by the reward model.
        """
        kwargs = {"padding": "max_length",
                  "truncation": True,
                  "max_length": 256,
                  "return_tensors": "pt"
                  }
        text = f"[INST] {prompt} [/INST] {response} </s>"
        inputs = self.rm_tokenizer(text, **kwargs).to(self.device)
        with torch.inference_mode():
            outputs = self.rm_model(**inputs)
        score = outputs['logits'].item()
        return score

    def generate_n_texts(self, model, question: str, n: int):
        """
        Generates `n` responses for a given question using the specified model.

        Args:
            model: The model to use for text generation.
            question (str): The input question.
            n (int): The number of responses to generate.

        Returns:
            list: A list of generated responses.
        """
        generated_texts = [model.generate(question) for _ in range(n)]
        return generated_texts

    def rejection_sampling(self, df, model, n):
        """
        Performs rejection sampling on a dataframe of questions, generating `n` responses for each 
        question and selecting the best based on reward model scores.

        Args:
            df (pd.DataFrame): The dataframe containing questions.
            model: The model to use for generating responses.
            n (int): The number of responses to generate for each question.

        Returns:
            pd.DataFrame: A dataframe containing the original questions and the selected best answers.
        """
        rejection_sampling_df = pd.DataFrame(df['Questions'])
        for i in range(len(df)):
            question = df.iloc[i]['Questions']
            generated_texts = self.generate_n_texts(model, question, n)
            best_answer = max(generated_texts, key=lambda response: self.rm_score(question, response))
            rejection_sampling_df.loc[i, 'Answer'] = best_answer
        return rejection_sampling_df

    def create_rejection_sampling_df(self, load_path, save_path):
        """
        Creates and saves a dataframe of selected answers after performing rejection sampling.

        Args:
            load_path (str): The file path for the input CSV file.
            save_path (str): The file path to save the rejection sampling dataframe.

        Returns:
            None
        """
        df = pd.read_csv(load_path)
        df = df[:5]
        rejection_sampling_df = self.rejection_sampling(df, self.sft_model, self.num_samples)

        # Rename the Questions column
        rejection_sampling_df.rename(columns={'Questions': 'Question'}, inplace=True)

        # Remove last sentences that contain 'tokens' or 'token'
        rejection_sampling_df['Answer'] = rejection_sampling_df['Answer'].apply(clean_text)
        
        rejection_sampling_df.to_csv(save_path, index=False)
    
    def formatting_prompts_func(self, examples):
        """
        Formats the prompts for rejection sampling, combining questions and selected best answers.

        Args:
            examples (dict): A dictionary containing 'Question' and 'Answer' keys.

        Returns:
            list: A list of formatted prompt strings.
        """
        return [f"[INST] {q} [/INST] {a} </s>" for q, a in zip(examples['Question'], examples['Answer'])]

    def load_and_process_data(self, Rejection_sampling_raw_path, save_path):
        """
        Loads and processes rejection sampling data from a CSV file, computes reward scores, 
        and saves the processed dataframe.

        Args:
            rejection_sampling_raw_path (str): The file path for the raw rejection sampling data CSV.
            save_path (str): The file path to save the processed rejection sampling dataframe.

        Returns:
            pd.DataFrame: The processed dataframe.
        """
        # The df in Rejection_sampling_df_path should be verfied manually to ensure that there are no 'token' or 'tokens' words in the generated answers
        rejection_sampling_df = pd.read_csv(Rejection_sampling_raw_path)

        # Formatting DataFrame
        rejection_sampling_df['text'] = self.formatting_prompts_func(rejection_sampling_df)

        #### Compute reward score for rs model
        rewards = []
        for question_and_answer in rejection_sampling_df['text']:
            instructions = self.sft_tokenizer.encode_plus(question_and_answer, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.rm_model(**instructions)

            logits = outputs[0].item()
            rewards.append(logits)
        rejection_sampling_df["rewards"] = rewards
        rejection_sampling_df = rejection_sampling_df.sort_values(by="rewards", ascending=True)

        # Take answers with high scores
        rejection_sampling_df = rejection_sampling_df[rejection_sampling_df['rewards'] > 5]
        rejection_sampling_df.to_csv(save_path, index=False)
        return rejection_sampling_df

    def convert_to_hf_datasets(self, df):
        """
        Converts the processed rejection sampling dataframe into Huggingface `Dataset` objects.

        Args:
            df (pd.DataFrame): The processed dataframe.

        Returns:
            tuple: A tuple containing:
                - train: Huggingface `Dataset` object for training.
                - val: Huggingface `Dataset` object for validation.
        """
        # Splitting and reset index to train and val
        train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)

        # Convert to Huggingface Datasets format
        train = Dataset.from_pandas(train_df)
        val = Dataset.from_pandas(val_df)

        # select the text column(s) 
        train = train.select_columns(['text'])

        # select the text column(s) 
        val = val.select_columns(['text'])
        return train, val

