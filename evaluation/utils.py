import pandas as pd
import torch

def answer(model, tokenizer, question, device, CoT = None):
    """
    Generates an answer to a given input question using a language model.

    Parameters:
        model (torch.nn.Module): The pre-trained language model to use for generating the answer.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer associated with the model.
        question (str): The input question or text prompt to generate an answer for.
        device (torch.device): The device (CPU or GPU) on which the model is running.
        CoT (str, optional): A "Chain-of-Thought" prompt to guide the model's reasoning. Defaults to None.

    Returns:
        str: The generated answer text.
    """
    # We omit <s> at the begining of the text since the tokenizer adds it
    if CoT:
        # text = f"<s>[INST] <<SYS>>\nYour answers should be concise and informative, containing no more than 256 tokens. {CoT}\n<</SYS>>\n\n{question} [/INST]"
        text = f"<s>[INST] <<SYS>>\nYour answers should be no longer than 256 tokens.\nDo not write how many tokens have been generated.\n{CoT}\n<</SYS>>\n\n{question} [/INST]"

    else:
        # text = f"<s>[INST] <<SYS>>\nYour answers should be concise and informative, containing no more than 256 tokens.\n<</SYS>>\n\n{question} [/INST]"
        text = f"<s>[INST] <<SYS>>\nYour answers should be no longer than 256 tokens.\nDo not write how many tokens have been generated.\n<</SYS>>\n\n{question} [/INST]"

    inputs = tokenizer(text, return_tensors="pt").to(device)
    inputs_length = len(inputs["input_ids"][0])
    with torch.inference_mode():
        # outputs = model.generate(**inputs, max_new_tokens=256, repetition_penalty=1.2, temperature=0.8)
        outputs = model.generate(**inputs, max_new_tokens=256)


    return tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)

def process_tagging_data(paths, source_df):
    """
    Processes multiple tagging data CSV files and combines them into a single DataFrame.

    Parameters:
        paths (list of str): A list of file paths to the tagging data CSV files.
        source_df (pd.DataFrame): A DataFrame containing the source information related to the questions.

    Returns:
        pd.DataFrame: A DataFrame containing combined tagging data, including the question, the preferred model,
                      and the preference strength.
    """
    tagging_dfs = []
    for path in paths:
        tagging_df = get_winner_model(path, source_df)
        tagging_df = tagging_df[['Question', 'Preference Strength', 'winner_model']]
        tagging_dfs.append(tagging_df)

    # Concatenate the DataFrames and return the combined DataFrame
    combined_df = pd.concat(tagging_dfs, ignore_index=True)
    return combined_df[['Question', 'winner_model', 'Preference Strength']]

def get_winner_model(path, source_df):
    """
    Determines the winner model for each question based on the tagging data and source information.

    Parameters:
        path (str): The file path to the tagging data CSV file.
        source_df (pd.DataFrame): A DataFrame containing the source information for the options.

    Returns:
        pd.DataFrame: A DataFrame with the question, chosen and rejected options, preference strength,
                      and the corresponding winner model.
    """
    df1 = pd.read_csv(path)
    df1.rename(columns={'Questions': 'Question'}, inplace=True)
    df2 = source_df
    merged_df = pd.merge(df1[['Question', 'Chosen', 'Rejected', 'Preference Strength']], df2[['Question', 'option1', 'option2', 'Source1', 'Source2']], how='inner', on='Question')        
    merged_df['winner_model'] = merged_df.apply(get_source, axis=1)
    merged_df = merged_df[['Question', 'Chosen', 'Rejected', 'Preference Strength', 'winner_model']]
    return merged_df

def get_source(row):
    """
    Determines the source model corresponding to the chosen option in a row.

    Parameters:
        row (pd.Series): A row of the DataFrame containing the chosen option and source information.

    Returns:
        str: The source model corresponding to the chosen option, or None if no match is found.
    """
    if row['Chosen'] == row['option1']:
        return row['Source1']
    elif row['Chosen'] == row['option2']:
        return row['Source2']
    else:
        return None
