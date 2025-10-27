# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import os
from typing import Optional
import openai
import google.generativeai as genai
import transformers
import torch
from huggingface_hub import hf_hub_download
from datetime import datetime, timedelta

def load_system_prompt(repo_id: str, filename: str) -> str:
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(file_path, 'r') as file:
        system_prompt = file.read()
    today = datetime.today().strftime('%Y-%m-%d')
    yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    model_name = repo_id.split("/")[-1]
    return system_prompt.format(name=model_name, today=today, yesterday=yesterday)


class OpenAIAPI:
    def __init__(self, model: str = "", llm_url: str = "http://localhost:8080/v1", system_prompt: Optional[str] = None):
        if model != "":  # OpenAI served model
            self.llm_client = openai.OpenAI(api_key=os.environ["OPENAI_KEY"])
        else:  # self served model
            self.llm_client = openai.OpenAI(api_key="UNUSED", base_url=llm_url)
        self.system_prompt = [{"role": "system", "content": system_prompt or "You are an AI assistant that helps people find information."}]
        self.model = model

    def __call__(self, prompt):
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=self.system_prompt + [{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content


class GoogleAPI:
    def __init__(self, model: str = "gemini-1.5-flash", system_prompt: Optional[str] = None):
        genai.configure(api_key=os.environ["GEMINI_KEY"])
        self.model = genai.GenerativeModel(model)
        self.system_prompt = system_prompt or "You are an AI assistant that helps people find information."

    def __call__(self, prompt):
        response = self.model.generate_content(f"{self.system_prompt}\n\n{prompt}")
        return response.text
    
    def list_models(self):
        return [model.name for model in genai.list_models() if "generateContent" in model.supported_generation_methods]


class LLamaAPI:
    def __init__(self, model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", system_prompt: Optional[str] = None):
        self.model = transformers.pipeline(
            "text-generation",
            model=model,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            return_full_text=False
        )
        self.system_prompt = [{"role": "system", "content": system_prompt or "You are an AI assistant that helps people find information."}]

    def __call__(self, prompt):
        chat_t = self.model.tokenizer.apply_chat_template(
            self.system_prompt + [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
        response = self.model(
            chat_t,
            max_new_tokens=1024,
        )

        return response[0]["generated_text"]


class LLMAPI:
    def __init__(
            self,
            api_name: str,
            system_prompt: Optional[str] = None,
            model: str = "gemini-1.5-flash",
            url: str = "http://localhost:8080/v1",
        ):
        if api_name == "openai":
            self.llm_api = OpenAIAPI(model, url, system_prompt)
        elif api_name == "google":
            self.llm_api = GoogleAPI(model, system_prompt)
        elif api_name == "transformers":
            self.llm_api = LLamaAPI(model, system_prompt)
    
    def __call__(self, prompt):
        return self.llm_api(prompt)
