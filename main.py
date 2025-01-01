from config import (
    connect_to_huggingface,
    connect_to_neptune,
    base_model_name,
    sft_training_args,
    sft_lora_config,
    rm_training_args,
    rm_lora_config,
    rs_training_args,
    rs_lora_config,
    DEVICE)
from data_processing import SFTProcessor, RewardModelProcessor, RejectionSamplingProcessor
from trainers import SFT, RewardModel
from utils import print_avg_length, print_number_of_trainable_model_parameters

def main():

    connect_to_neptune()
    connect_to_huggingface()

    # Preprocess SFT data
    sft_processor = SFTProcessor(base_model_name)
    df = sft_processor.load_and_process_data("./data/GPT_Answers.csv", "./data/sft_model.csv")
    train, val, train_df, val_df, test_df = sft_processor.convert_to_hf_datasets(df)

    # List of dataframes and their labels
    dataframes = {
        'Data': df,
        'Train': train_df,
        'val': val_df,
        'test': test_df
    }

    for label, dataframe in dataframes.items():
        print(f"{label} stats:")
        print_avg_length(dataframe, 'Question_list', "Average question length:")
        print_avg_length(dataframe, 'Answer_list', "Average answer length:")
        print_avg_length(dataframe, 'text_list', "Average text length:")
        print("\n----------------------------------")
    
    # Train SFT Model
    sft = SFT(base_model_name, sft_training_args)
    print_number_of_trainable_model_parameters(sft.model, "Intial number of parameters:")
    sft_trainer = sft.fit(train, val, sft_lora_config)
    print_number_of_trainable_model_parameters(sft.model, "Number of parameters after lora peft:")
    sft.train(sft_trainer)
    # sft.merge_and_push_model()

    # Preprocess rm data
    rm_processor = RewardModelProcessor(base_model_name)
    df = rm_processor.load_and_process_data("/sise/home/odedreg/rm_dataset.csv")
    train, val = rm_processor.convert_to_hf_datasets(df)

    # Train Reward Model
    rm = RewardModel(base_model_name, rm_training_args)
    print_number_of_trainable_model_parameters(rm.model, "Intial number of parameters:")
    rm_trainer = rm.fit(train, val, rm_lora_config)
    print_number_of_trainable_model_parameters(rm.model, "Number of parameters after lora peft:")
    rm.train(rm_trainer)
    
    # Preprocess Rejection Sampling data
    sft_model_name = f"odedregev/{sft_training_args.output_dir}"
    new_model_name = f"/sise/home/odedreg/{rm_training_args.output_dir}/"
    rs_raw_path = "rejection_sampling_raw_df.csv"
    num_samples = 4
    rs_processor = RejectionSamplingProcessor(sft_model_name, base_model_name, new_model_name, num_samples, device=DEVICE)
    rs_processor.create_rejection_sampling_df("/sise/home/odedreg/rs_questions.csv", rs_raw_path)
    df = rs_processor.load_and_process_data(f"/sise/home/odedreg/{rs_raw_path}", "rejection_sampling_final_df")
    train, val = rm_processor.convert_to_hf_datasets(df)

    # Train Rejection Sampling model using Supervised Fine-Tuning
    rs = SFT(sft_model_name, rs_training_args)
    print_number_of_trainable_model_parameters(rs.model, "Intial number of parameters:")
    rs_trainer = rs.fit(train, val, rs_lora_config)
    print_number_of_trainable_model_parameters(rs.model, "Number of parameters after lora peft:")
    rs.train(rs_trainer)
    rs.merge_and_push_model()

if __name__ == "__main__":
    main()
