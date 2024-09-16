import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


class SnowflakeEmbed:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Snowflake/snowflake-arctic-embed-m-long"
        )
        self.model = AutoModel.from_pretrained(
            "Snowflake/snowflake-arctic-embed-m-long",
            trust_remote_code=True,
            add_pooling_layer=False,
            safe_serialization=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, text: str):
        tokens = self.tokenizer(
            [text], padding=True, truncation=True, return_tensors="pt", max_length=2048
        )
        tokens = tokens.to(self.device)
        embeddings = self.model(**tokens)[0][:, 0]
        return embeddings.detach().cpu().squeeze(0).numpy()


class NvEmbedE5:
    def __init__(self):
        self.api_key = os.environ.get("NVIDIA_API_KEY", None)
        if self.api_key is None:
            raise ValueError("No NVIDIA API key set!")

        from openai import OpenAI

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://integrate.api.nvidia.com/v1",
        )

    def __call__(self, text: str):
        response = self.client.embeddings.create(
            input=[text],
            model="nvidia/nv-embedqa-e5-v5",
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "END"},
        )
        return np.array(response.data[0].embedding, dtype="float32")
