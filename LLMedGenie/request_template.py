from pydantic import BaseModel


class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: float = 100
    temperature: float = 0.7
    top_k: float = 50
    top_p: float = 0.95
    num_seq: int = 1
    output_file: str = None
