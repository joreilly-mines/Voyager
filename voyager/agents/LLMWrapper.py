from typing import Optional, List, Mapping, Any
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain.chat_models.base import BaseChatModel
from pydantic import Field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)

class TransformersLLM(BaseChatModel):
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    temperature: float = 0.7
    max_new_tokens: int = 512

    # Internal fields (not part of the LLM schema)
    model: Any = Field(default=None, exclude=True)
    tokenizer: Any = Field(default=None, exclude=True)
    device: Any = Field(default=None, exclude=True)

    def __init__(self, **data):
        super().__init__(**data)  # initialize pydantic fields first

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16).to(self.device) #'''quantization_config=bnb_config, device_map="auto"'''
        self.model.eval()

    @property
    def _llm_type(self) -> str:
        return "HF-Transformer-model"

    def _format_prompt(self, messages: List[BaseMessage]) -> str:
        prompt = ""
        for msg in messages:
            if isinstance(msg, SystemMessage):
                prompt += f"<|system|>\n{msg.content}\n"
            elif isinstance(msg, HumanMessage):
                prompt += f"<|user|>\n{msg.content}\n"
            elif isinstance(msg, AIMessage):
                prompt += f"<|assistant|>\n{msg.content}\n"
        prompt += "<|assistant|>\n"
        return prompt

    def _generate_response(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id
        )
        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return output_text[len(prompt):].strip()

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any
    ) -> AIMessage:
        prompt = self._format_prompt(messages)
        response = self._generate_response(prompt)
        return AIMessage(content=response)

    def get_num_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))



'''
from typing import List, Optional, Mapping, Any
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain.chat_models.base import BaseChatModel

from pydantic import PrivateAttr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class TransformersLLM(BaseChatModel):
    _device: torch.device = PrivateAttr()
    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()


    def __init__(self, 
                 model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
                 temperature: float = 0.7,
                 max_new_tokens: int = 512, 
                 device: Optional[str] = None):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        self._model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(self._device)
        self._model.eval()

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        #generator = pipeline("text-generation", model=self.model, tokenizer = tokenizer, device = self.device)

        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name

    @property
    def _llm_type(self) -> str:
        return "HF-Transformer-model"

    def _format_prompt(self, messages: List[BaseMessage]) -> str:
        prompt = ""
        for msg in messages:
            if isinstance(msg, SystemMessage):
                prompt += f"<|system|>\n{msg.content}\n"
            elif isinstance(msg, HumanMessage):
                prompt += f"<|user|>\n{msg.content}\n"
            elif isinstance(msg, AIMessage):
                prompt += f"<|assistant|>\n{msg.content}\n"
        prompt += "<|assistant|>\n"
        return prompt

    def _generate_response(self, prompt: str) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        output = self._model.generate(**inputs,
                                     max_new_tokens=self.max_new_tokens,
                                     temperature=self.temperature,
                                     do_sample=self.temperature > 0,
                                     pad_token_id=self.tokenizer.eos_token_id)
        output_text = self._tokenizer.decode(output[0], skip_special_tokens=True)
        return output_text[len(prompt):].strip()
    
    def _generate(self,
                  messages: List[BaseMessage],
                  stop: Optional[List[str]] = None,
                  run_manager: Optional[Any] = None,
                  **kwargs: Any,) -> AIMessage:
        prompt = self._format_prompt(messages)
        response = self._generate_response(prompt)
        return AIMessage(content=response)
    
    def get_num_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

'''