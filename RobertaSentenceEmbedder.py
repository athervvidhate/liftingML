import os
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizerFast, RobertaModel, RobertaForMaskedLM

class RobertaSentenceEmbedder:
    def __init__(self, model_name="./roberta_finetuned", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, sentences, batch_size=32, normalize=True):
        if isinstance(sentences, str):
            sentences = [sentences]
        
        all_embeddings = []
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start:start+batch_size]

            encoded_input = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded_input)
                token_embeddings = outputs.last_hidden_state
                attention_mask = encoded_input['attention_mask']

                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
                sum_embeddings = (token_embeddings * input_mask_expanded).sum(1)
                sum_mask = attention_mask.sum(1).unsqueeze(-1)
                embeddings = sum_embeddings / sum_mask

                if normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)
    
    def save(self, save_directory):
        """
        Save the model and tokenizer to the specified directory.
        """
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def load(cls, load_directory, device=None):
        """
        Load the model and tokenizer from the specified directory.
        """
        return cls(model_name=load_directory, device=device)