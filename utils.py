import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import torch
from peft import PeftModel

def plot_cumulative_text_length(df):
    """
    Plots a graph of cumulative count of rows based on text length.

    Args:
        df (DataFrame): DataFrame containing 'text_length' and 'cumulative_count' columns.
    """
    plt.plot(df['text_length'], df['cumulative_count'], marker='o')
    plt.xlabel('Text Length')
    plt.ylabel('Cumulative Count of Rows')
    plt.title('Cumulative Count of Rows Until Text Length')
    plt.grid(True)
    plt.show()

def plot_dataset_histogram(df):
    """
    Plots a histogram of dataset counts from the 'Dataset' column.

    Args:
        df (DataFrame): DataFrame containing 'Dataset' column.
    """
    plt.figure(figsize=(5, 4))
    df['Dataset'].value_counts().plot(kind='bar', color='brown')
    plt.title('Histogram of Dataset')
    plt.xlabel('Dataset')
    plt.ylabel('Number of Q/A')
    plt.xticks(rotation=45)
    plt.show()

def print_avg_length(df, col_name, text):
    """
    Prints the average length of the specified column in the DataFrame.

    Args:
        df (DataFrame): DataFrame containing the data.
        col_name (str): Name of the column for which average length is to be calculated.
        text (str): Text to display along with the average length.

    Returns:
        None
    """
    col_lengths = df[col_name].apply(len)
    col_avg_lengths = col_lengths.mean()
    print(text, int(col_avg_lengths))

def print_number_of_trainable_model_parameters(model, text=""):
    """
    Prints the number of trainable parameters and the total number of parameters in the model.

    Args:
        model (nn.Module): The model to analyze.
        text (str, optional): Optional text to prepend to the output. Defaults to "".

    Returns:
        None
    """
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
            
    trainable_model_params_str  = "{:,}".format(trainable_model_params)
    all_model_params_str  = "{:,}".format(all_model_params)
    
    print(f"""{text}
          Trainable model parameters: {trainable_model_params_str}\nAll model parameters: {all_model_params_str}\nPercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%\n""")

def clean_text(text):
    """
    Cleans the input text by removing the last part if it contains the word 'tokens' or 'token'.

    This function checks if the input text contains any newline characters. If it does, 
    it splits the text into parts using the newline character as a delimiter. It then 
    examines the last part of the split text. If the last part contains the word 'tokens' 
    or 'token' (case insensitive), it removes this part and returns the rest of the text. 
    Otherwise, it returns the original text.

    Parameters:
    text (str): The input text to be cleaned.

    Returns:
    str: The cleaned text with the last part removed if it contains 'tokens' or 'token', 
         otherwise the original text.
    """
    if '\n' in text:
        parts = text.split('\n')
        last_part = parts[-1]

        if ('tokens' in last_part.lower()) or ('token' in last_part.lower()):
            return '\n'.join(parts[:-1])  # Return the text without the last part
    return text

def read_model_and_tokenizer(model_name):
    """
    Creates a language model and tokenizer from the specified model name.

    Args:
        model_name (str): The path of the pre-trained model.

    Returns:
        tuple: A tuple containing the language model and tokenizer.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_safetensors=True,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

def read_rm_and_tokenizer(base_model_name, new_model_name):
    """
    Reloads the base model in FP16 precision and merges it with the LoRA weights.

    Args:
        base_model_name (str): The name of the base model.
        new_model_name (str): The name of the new model with LoRA weights.

    Returns:
        tuple: A tuple containing the model with merged LoRA weights and the tokenizer.
    """
    load_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="auto",
            num_labels=1,
        )


    rm_model = PeftModel.from_pretrained(load_model, new_model_name)

    # Reload tokenizer to save it
    rm_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    rm_tokenizer.pad_token = rm_tokenizer.eos_token
    rm_tokenizer.padding_side = "right"
    return rm_model, rm_tokenizer
