#%%
# visualize_data.py
from config import connect_to_huggingface, connect_to_neptune, base_model_name
from data_processing import DataProcessor
from utils import plot_cumulative_text_length, plot_dataset_histogram

def main():
    connect_to_huggingface()

    data_processor = DataProcessor(base_model_name)
    df, _, _, _ = data_processor.load_and_process_data(
        "/sise/home/odedreg/GPT_Answers.csv",
        "/sise/home/odedreg/sft_model.csv"
    )

    plot_cumulative_text_length(df)
    plot_dataset_histogram(df)

if __name__ == "__main__":
    main()