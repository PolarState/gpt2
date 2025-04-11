import argparse
from datetime import datetime
import torch
import transformers
import os
import logging

from cfg import cfg_defines, cfg_datasets

from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer


parser = argparse.ArgumentParser(
    description="Options for main",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# parser.add_argument("-o", "--output", required=True, help="Path to output folder.")
parser.add_argument(
    "-r",
    "--resume",
    default=False,
    help="Resume training.",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "-m",
    "--model",
    default="GPT2",
    help="name of the model to use. 'GPT2' and 'GPTNeoX' are the only supported options currently.",
)

args = parser.parse_args()
config = vars(args)
resume = config["resume"]
model_name = config["model"]
# output_path = config["output"]


cfg = cfg_defines.cfg3b
cfg_start_symbol = "22"


# use cpu or gpu based on your system
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

if model_name == "GPT2":
    gpt_config = transformers.GPT2Config()
    model = transformers.GPT2LMHeadModel(gpt_config)
    batch_size = 22
    output_dir = ("gpt2-cfg3b/polm-0/",)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

elif model_name == "GPTNeoX":
    # Initalize GPTNeoX to use the same tokenizer, number of heads etc as GPT2
    gpt_config = transformers.GPTNeoXConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=512,
    )
    model = transformers.GPTNeoXForCausalLM(gpt_config)
    batch_size = 9
    output_dir = "gptneox-cfg3b/polm-1/"
    tokenizer = transformers.GPTNeoXTokenizerFast.from_pretrained(
        "openai-community/gpt2"
    )
else:
    raise NotImplementedError(f"no implementation for {model_name}")


training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",
    eval_steps=5000,
    num_train_epochs=1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=0.003,
    lr_scheduler_type="linear",
    adam_beta1=0.9,
    adam_beta2=0.98,
    weight_decay=0.01,
    logging_strategy="steps",
    logging_steps=20000,
    save_steps=20000,
    save_total_limit=10,
    report_to="wandb",
    dataloader_pin_memory=False,
)


data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=cfg_datasets.CFGRandomGenerationDataset(
        cfg, cfg_start_symbol, 100000 * 96 * 512, tokenizer=tokenizer, device=device
    ),
    eval_dataset=cfg_datasets.CFGRandomGenerationDataset(
        cfg, cfg_start_symbol, 10000 * 512, tokenizer=tokenizer, device=device
    ),
    data_collator=data_collator,
)

try:
    checkpoint_names = os.listdir(output_dir)
    if resume:
        last_checkpoint = list(reversed(sorted(checkpoint_names)))[0]
        trainer.train(f"{output_dir}/{last_checkpoint}/")
    else:
        trainer.train()
except Exception as e:
    print(f"AT: {datetime.now()}")
    logging.exception("training exited with exception.")
    print(f"GPU temp: {torch.cuda.temperature()}")
