import argparse
from datetime import datetime
import torch
import transformers
import os
import logging
import psutil
import subprocess

from cfg import cfg_defines, cfg_datasets
from cfg.cfg_grammar import CFGrammar
from cfg.cfg_tokenizers import CFGCharacterTokenizer
from cfg.hf_adapter import HFTokenizerAdapter

from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer, TrainerCallback

DATASET_TRAIN_PATH = "../CFG/datasets/cfg3b_train_dataset.bin"
DATASET_VALIDATION_PATH = "../CFG/datasets/cfg3b_val_dataset.bin"

logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def get_gpu_temp():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return int(result.stdout.strip())
    except Exception:
        return -1


class MetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not torch.cuda.is_available():
            return
        metrics = {
            "system/gpu_mem_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "system/gpu_mem_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "system/gpu_temp_c": get_gpu_temp(),
            "system/ram_used_gb": psutil.virtual_memory().used / 1024**3,
            "system/ram_percent": psutil.virtual_memory().percent,
        }
        logging.info(f"system_metrics: {metrics}")
        if logs is not None:
            logs.update(metrics)

parser = argparse.ArgumentParser(
    description="Options for main",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("-o", "--output", default=None, help="Path to output folder.")
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
parser.add_argument(
    "--reverse",
    default=False,
    help="Read the dataset in reverse window order (last window first).",
    action=argparse.BooleanOptionalAction,
)

args = parser.parse_args()
config = vars(args)
resume = config["resume"]
model_name = config["model"]
output_path = config["output"]
reverse = config["reverse"]


cfg = cfg_defines.cfg3b
cfg_start_symbol = "22"

grammar = CFGrammar.from_name("cfg3b")
tokenizer = HFTokenizerAdapter(CFGCharacterTokenizer(vocab=grammar.terminal_symbols))

# use cpu or gpu based on your system
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.95)

if model_name == "GPT2":
    gpt_config = transformers.GPT2Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = transformers.GPT2LMHeadModel(gpt_config)
    batch_size = 96
    output_dir = output_path or "gpt2-cfg3b/polm-0/"

elif model_name == "GPTNeoX":
    gpt_config = transformers.GPTNeoXConfig(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
    )
    model = transformers.GPTNeoXForCausalLM(gpt_config)
    batch_size = 96
    output_dir = output_path or "gptneox-cfg3b/polm-5/"

else:
    raise NotImplementedError(f"no implementation for {model_name}")


training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",
    eval_steps=5000,
    num_train_epochs=1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=0.0003,
    lr_scheduler_type="linear",
    adam_beta1=0.9,
    adam_beta2=0.98,
    weight_decay=0.1,
    logging_strategy="steps",
    logging_steps=20000,
    save_steps=20000,
    save_total_limit=10,
    report_to="wandb",
    dataloader_pin_memory=False,
    fp16=True,
)

# train_dataset=cfg_datasets.CFGRandomGenerationDataset(
#     cfg, cfg_start_symbol, 100000 * 96 * 512, tokenizer=tokenizer, device=device
# ),
# eval_dataset=cfg_datasets.CFGRandomGenerationDataset(
#     cfg, cfg_start_symbol, 10000 * 512, tokenizer=tokenizer, device=device
# ),
train_dataset=cfg_datasets.CFGFileDataset(
        filename=DATASET_TRAIN_PATH, device=device, reverse=reverse
    )
eval_dataset=cfg_datasets.CFGFileDataset(
        filename=DATASET_VALIDATION_PATH, device=device, reverse=reverse
    )
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[MetricsCallback],
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
