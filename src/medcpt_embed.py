import torch
from transformers import AutoTokenizer, AutoModel

class MedCPTEncoder:
    def __init__(self, model_name: str, device: str = None):
        self.device = device or ("cuda:1" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=True,
        ).to(self.device).eval()

    @torch.no_grad()
    def encode(self, texts: list[str], max_length: int = 512):
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**batch)
        emb = outputs.last_hidden_state[:, 0]
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy()