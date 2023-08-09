
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

import torch
import transformers
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizerFast)

LlamaModelWeight = Literal["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-7b-hf"]
Role = Literal["system", "user", "assistant"]

class ModelResolution(str, Enum):
    i4 = 'int4'
    i8 = 'int8'
    f16 = 'float16'
    f32 = 'float32'

class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]



class LlamaModel():
    B_INST, E_INST = "[INST]", "[/INST]"
    BOS, EOS = '<s>', '</s>'
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    DEFAULT_SYSTEM_PROMPT = """\
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
    def __init__(self, 
                 model_name: LlamaModelWeight, 
                 model_resolution: ModelResolution | None, 
                 device_map: str = 'auto', 
                 use_tf_core: bool | None = True):

        if use_tf_core is not None:
            torch.backends.cudnn.allow_tf32 = use_tf_core
            torch.backends.cuda.matmul.allow_tf32 = use_tf_core
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = use_tf_core
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = use_tf_core
        
        self.tokenizer: LlamaTokenizerFast = AutoTokenizer.from_pretrained(model_name)
        self.generation_config = GenerationConfig.from_pretrained(model_name)

        # self.tokenizer.pad_token_id = -1

        load_config: Dict[str, Any] = {
            'device_map': device_map 
        }

        # Set quantisation / bit width of model parameters
        bnb_config=None
        match model_resolution:
            case ModelResolution.i4:
                bnb_config = transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            case ModelResolution.i8:
                bnb_config = transformers.BitsAndBytesConfig(
                    load_in_8bit=True
                )
            case ModelResolution.f16:
                load_config['torch_dtype'] = torch.float16
            case ModelResolution.f32:
                load_config['torch_dtype'] = torch.float32
            
        self.model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            **load_config)
        
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.model.eval()

    def tokenize(self, text: str | List[str]):
        token_dict = self.tokenizer(text, add_special_tokens=True, padding=True, return_tensors='pt')
        return token_dict['input_ids']

    def generate(self, text: str) -> str:
        tokens = self.tokenizer(text, add_special_tokens=True, padding=True, return_tensors='pt')

        output = self.model.generate(
            tokens['input_ids'].to(self.model.device),
            generation_config=self.generation_config,
            # max_new_tokens=128,
            # do_sample=True,
            # top_k=10,
            num_return_sequences=1
        )
        
        return self.tokenizer.decode(output[0])
    
class Llama2ChatModel(LlamaModel):
    def __init__(self, 
                 model_name: LlamaModelWeight, 
                 model_resolution: ModelResolution | None, 
                 device_map: str = 'auto', 
                 use_tf_core: bool | None = True):
        super().__init__(model_name=model_name, model_resolution=model_resolution, device_map=device_map, use_tf_core=use_tf_core)

    def preprocess_dialog(self, dialogs: List[Dialog], tokenize: bool = False):
        all_dialog = []
        for dialog in dialogs:
            if dialog[0]["role"] != "system":
                dialog = [
                    {
                        "role": "system",
                        "content": self.DEFAULT_SYSTEM_PROMPT,
                    }
                ] + dialog
            dialog = [
                {
                    "role": dialog[1]["role"],
                    "content": self.B_SYS
                    + dialog[0]["content"]
                    + self.E_SYS
                    + dialog[1]["content"],
                }
            ] + dialog[2:]
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )

            dialog_tokens = [f"{self.BOS}{self.B_INST} {(prompt['content']).strip()} {self.E_INST} {(answer['content']).strip()} {self.EOS}"
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ]
            dialog_tokens += [f"{self.BOS}{self.B_INST} {(dialog[-1]['content']).strip()} {self.E_INST}"]

            dialog_str = '\n'.join(dialog_tokens)
            all_dialog.append(dialog_str)

        return all_dialog





