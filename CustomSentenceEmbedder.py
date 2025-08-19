import os
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizerFast, RobertaModel
from transformers import AlbertTokenizerFast, AlbertModel
import gcsfs
import tempfile
import shutil

class CustomSentenceEmbedder:
    def __init__(self, model_name="./roberta_finetuned", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        # Check if model_name is a GCS path
        if model_name.startswith("gs://"):
            # Load from Cloud Storage
            self._load_from_gcs(model_name)
        else:
            # Load from local storage
            self._load_from_local(model_name)
        
        self.model.eval()

    def _load_from_gcs(self, gcs_path):
        """Load model from Google Cloud Storage"""
        try:
            # Create a temporary directory
            temp_dir = tempfile.mkdtemp()
            print(f"Downloading model to temporary directory: {temp_dir}")
            
            # Download model files from GCS
            fs = gcsfs.GCSFileSystem()
            
            # List all files in the GCS directory
            gcs_files = fs.ls(gcs_path)
            
            # Download each file
            for gcs_file in gcs_files:
                if fs.isfile(gcs_file):
                    # Get the filename
                    filename = os.path.basename(gcs_file)
                    local_path = os.path.join(temp_dir, filename)
                    
                    # Download the file
                    with fs.open(gcs_file, 'rb') as remote_file:
                        with open(local_path, 'wb') as local_file:
                            shutil.copyfileobj(remote_file, local_file)
                    
                    print(f"Downloaded: {filename}")
            
            if 'albert_finetuned' in gcs_path:
                self.tokenizer = AlbertTokenizerFast.from_pretrained(temp_dir)
                self.model = AlbertModel.from_pretrained(
                    temp_dir,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                ).to(self.device)
            else:
                self.tokenizer = RobertaTokenizerFast.from_pretrained(temp_dir)
                self.model = RobertaModel.from_pretrained(
                    temp_dir,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                ).to(self.device)
            
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
            print("Model loaded successfully from GCS")
            
        except Exception as e:
            raise Exception(f"Failed to load model from GCS {gcs_path}: {e}")

    def _load_from_local(self, local_path):
        """Load model from local storage"""
        try:
            # Use Albert if model_name is exactly './albert_finetuned'
            if local_path == "./albert_finetuned":
                self.tokenizer = AlbertTokenizerFast.from_pretrained(local_path)
                self.model = AlbertModel.from_pretrained(
                    local_path,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                ).to(self.device)
            else:
                self.tokenizer = RobertaTokenizerFast.from_pretrained(local_path)
                self.model = RobertaModel.from_pretrained(
                    local_path,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                ).to(self.device)
        except Exception as e:
            raise Exception(f"Failed to load model from local path {local_path}: {e}")

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