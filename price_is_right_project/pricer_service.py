import modal
from pathlib import PurePosixPath

## Setup - define our infrastructure with code
app = modal.App("pricer-service")
secrets = [modal.Secret.from_name("huggingface-secret")]

image = modal.Image.debian_slim().pip_install(
    "huggingface", "torch", "transformers", "bitsandbytes", 
    "accelerate", "peft", "huggingface_hub[hf_transfer]"
).env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})

## This is where we cache model files to avoid redownloading each time a container is started
hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)

GPU = "T4"

## Keep N containers active to avoid cold starts
MIN_CONTAINERS = 0

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
PROJECT_NAME = "pricer"
HF_USER = "shayharding"
RUN_NAME = "2025-03-11_22.28.36"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
REVISION = "f0597ccf35e1cab4f4bf446dbb464fef2b021bae"
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"

## Mount for cache location
MODEL_DIR = PurePosixPath("/models")
BASE_DIR = MODEL_DIR / BASE_MODEL
FINETUNED_DIR = MODEL_DIR / FINETUNED_MODEL

QUESTION = "How much does this cost to the nearest dollar?"
PREFIX = "Price is $"

@app.cls(image=image, secrets=secrets, gpu=GPU, timeout=1800, min_containers=MIN_CONTAINERS, volumes={MODEL_DIR: hf_cache_vol})
class Pricer:
    @modal.enter()
    def setup(self):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        ## Download and cache model files to the volume
        snapshot_download(BASE_MODEL, local_dir=BASE_DIR)
        snapshot_download(FINETUNED_MODEL, revision=REVISION, local_dir=FINETUNED_DIR)

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_DIR)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_DIR, 
            quantization_config=quant_config,
            device_map="auto"
        )
    
        self.fine_tuned_model = PeftModel.from_pretrained(self.base_model, FINETUNED_DIR, revision=REVISION)

    ## Get price for a given product description
    @modal.method()
    def price(self, description: str) -> float:
        import re, torch
        from transformers import set_seed
    
        set_seed(42)
        prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        attention_mask = torch.ones(inputs.shape, device="cuda")
        outputs = self.fine_tuned_model.generate(inputs, attention_mask=attention_mask, max_new_tokens=5, num_return_sequences=1)
        result = self.tokenizer.decode(outputs[0])
    
        contents = result.split("Price is $")[1]
        contents = contents.replace(',','')
        match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
        return float(match.group()) if match else 0

