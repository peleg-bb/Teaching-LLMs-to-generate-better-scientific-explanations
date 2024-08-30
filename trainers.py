import torch
from transformers import AutoModelForCausalLM,AutoModelForSequenceClassification, BitsAndBytesConfig, AutoTokenizer
from trl import SFTTrainer, RewardTrainer
from peft import PeftModel
from abc import ABC, abstractmethod
from utils import read_model_and_tokenizer

class BaseTrainer(ABC):
    def __init__(self, model_name, training_args):
        """
        Initializes the base trainer with model name and training arguments.

        Args:
            model_name (str): The name of the pre-trained model.
            training_args (TrainingArguments): Arguments for training.
        """
        self.model_name = model_name
        self.training_args = training_args
        self.model, self.tokenizer = self.create_model_and_tokenizer()
        self.model.config.use_cache = False
        self.output_dir = self.training_args.output_dir

    @abstractmethod
    def create_model_and_tokenizer(self):
        pass

    @abstractmethod
    def fit(self, train, val, *args, **kwargs):
        pass

    @abstractmethod
    def train(self, trainer):
        pass

class SFT(BaseTrainer):
    def create_model_and_tokenizer(self):
        """
        Creates and returns a language model and tokenizer for SFT.

        Returns:
            tuple: A tuple containing the language model and tokenizer.
        """
        model, tokenizer = read_model_and_tokenizer(self.model_name)
        return model, tokenizer

    def fit(self, train, val, peft_config, dataset_text_field="text", max_seq_length=256):
        """
        Creates and returns an SFTTrainer instance.

        Args:
            train (Dataset): Training dataset.
            val (Dataset): Validation dataset.
            peft_config (LoraConfig): Configuration for LoRA.
            dataset_text_field (str, optional): The field in the dataset containing the text. Defaults to "text".
            max_seq_length (int, optional): Maximum sequence length. Defaults to 256.

        Returns:
            SFTTrainer: SFTTrainer instance.
        """
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train,
            eval_dataset=val,
            peft_config=peft_config,
            dataset_text_field=dataset_text_field,
            max_seq_length=max_seq_length,
            tokenizer=self.tokenizer,
            args=self.training_args,
            packing=False,
        )
        return trainer

    def train(self, trainer):
        """
        Trains the model using the given SFTTrainer instance and saves the model.

        Args:
            trainer (SFTTrainer): SFTTrainer instance.
        """
        trainer.train()

        # Save the model to the output directory
        trainer.save_model(self.output_dir)

    def merge_and_push_model(self):
        """
        Merges the base model with LoRA weights and pushes the final model to the hub.
        """
        load_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        model = PeftModel.from_pretrained(load_model, self.output_dir)
        model = model.merge_and_unload()

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model.push_to_hub(self.output_dir, use_temp_dir=False)
        tokenizer.push_to_hub(self.output_dir, use_temp_dir=False)


class RewardModel(BaseTrainer):
    def create_model_and_tokenizer(self):
        """
        Creates and returns the model and tokenizer for the Reward Model.

        Returns:
            tuple: A tuple containing the model and tokenizer.
        """
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            use_safetensors=True,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
            num_labels=1,
        )
        model.config.pad_token_id = model.config.eos_token_id 

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        return model, tokenizer

    def fit(self, train, val, peft_config, max_seq_length=256):
        """
        Fits the Reward Model using the provided training and validation data.

        Args:
            train (Dataset): Training dataset.
            val (Dataset): Validation dataset.
            peft_config (LoraConfig): Configuration for LoRA.
            max_seq_length (int, optional): Maximum sequence length. Defaults to 256.

        Returns:
            RewardTrainer: An instance of RewardTrainer.
        """
        trainer = RewardTrainer(model=self.model,
                        tokenizer=self.tokenizer,
                        train_dataset=train,
                        eval_dataset=val,
                        args=self.training_args,
                        peft_config =peft_config,
                        max_length=max_seq_length,
                        )
        return trainer

    def train(self, trainer):
        """
        Trains the Reward Model using the provided RewardTrainer instance and saves the model.

        Args:
            trainer (RewardTrainer): The RewardTrainer instance used for training the model.
        """

        trainer.train()
        trainer.save_model(self.output_dir)
