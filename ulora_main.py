import os
import json
import time
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from typing import Dict, List
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    GPT2PreTrainedModel,
    GPT2Tokenizer,
    GPT2Model,
    GPT2Config,
    TrainingArguments
)

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use non-interactive backend
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']  # Set font family (includes SimHei)
plt.rcParams['axes.unicode_minus'] = False  # Correctly display negative signs

# Multi-GPU mode: Do not set CUDA_VISIBLE_DEVICES, use standard PyTorch device management
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

# ====================================================================
# Configuration Management 
# ====================================================================

class FederatedConfig:
    """Define your configuration here as shown in config_example.py"""
    
    







    
    def get_data_paths(self, data_type):
        """Get data file paths for training or prediction"""
        if data_type == "train":
            return [os.path.join(self.base_data_dir, f) for f in self.train_files]
        elif data_type == "predict":
            return [os.path.join(self.base_data_dir, f) for f in self.predict_files]
        else:
            raise ValueError(f"Unknown data_type: {data_type}. Use 'train' or 'predict'")
    
    def get_output_paths(self):
        """Get output directory paths"""
        return {
            "base_dir": self.output_dir,
            "plots": os.path.join(self.output_dir, "plots"),
            "logs": os.path.join(self.output_dir, "logs"),
            "models": os.path.join(self.output_dir, "models")
        }
 

# ====================================================================
# Multi-GPU style efficient dataset class (avoid OOM issues)
# ====================================================================

class SolarDataset(Dataset):
    """Solar power forecasting dataset class, based on FedAvg_electric implementation"""
    def __init__(self, data_file, tokenizer, max_length=1024, label_scale=1.0, indices=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_scale = label_scale
        
        # Load data
        with open(data_file, 'r', encoding='utf-8') as f:
            all_samples = json.load(f)
        
        # If indices are provided, only use specified samples
        if indices is not None:
            self.dataset_samples = [all_samples[i] for i in indices]
            print(f"Loaded {len(indices)} specified samples from {data_file} (total {len(all_samples)} samples)")
        else:
            self.dataset_samples = all_samples
            print(f"Loaded {len(all_samples)} samples from {data_file}")
    
    def __len__(self):
        return len(self.dataset_samples)
    
    def __getitem__(self, idx):
        # Handle batch indexing case - DataLoader may pass a list
        if isinstance(idx, list):
            return [self._get_single_item(i) for i in idx]
        else:
            return self._get_single_item(idx)
    
    def _get_single_item(self, idx):
        sample = self.dataset_samples[idx]
        
        # Construct text prompt: instruction + input
        prompt = f"{sample['instruction']} {sample['input']}"
        
        # Tokenize text
        tokenized = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Process labels: extract prediction values from output
        # Format: "7633.8,7361.0,7136.0,..."
        output_values = [float(x.strip()) for x in sample['output'].split(",")]
        labels = torch.tensor(output_values, dtype=torch.float32) * self.label_scale
        
        # Save original labels (unscaled)
        original_labels = torch.tensor(output_values, dtype=torch.float32)
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': labels,
            'original_labels': original_labels,
            'instruction': sample['instruction'],
            'input': sample['input'],
            'output': sample['output']
        }
    
    @staticmethod
    def create_train_val_split(data_file, tokenizer, max_length=1024, label_scale=1.0, train_ratio=0.9, seed=42):
        """Create training and validation datasets, manually split to avoid Subset issues"""
        # Read data to determine total count
        with open(data_file, 'r', encoding='utf-8') as f:
            all_samples = json.load(f)
        
        # Manual index splitting
        total_samples = len(all_samples)
        train_size = int(train_ratio * total_samples)
        
        # Use fixed seed to ensure reproducibility
        np.random.seed(seed)
        indices = np.random.permutation(total_samples)
        
        train_indices = indices[:train_size].tolist()
        val_indices = indices[train_size:].tolist()
        
        # Create independent dataset objects
        train_dataset = SolarDataset(data_file, tokenizer, max_length, label_scale, train_indices)
        val_dataset = SolarDataset(data_file, tokenizer, max_length, label_scale, val_indices)
        
        return train_dataset, val_dataset

# Custom collate function for solar power data
def solar_collate_fn(batch):
    """Custom collate function to handle SolarDataset output"""
    # If batch is nested list (returned from __getitem__), need to flatten
    if isinstance(batch[0], list):
        flat_batch = []
        for item in batch:
            if isinstance(item, list):
                flat_batch.extend(item)
            else:
                flat_batch.append(item)
        batch = flat_batch
    
    # Merge batch data
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    original_labels = torch.stack([item['original_labels'] for item in batch])
    
    # Handle non-tensor fields
    instructions = [item['instruction'] for item in batch]
    inputs = [item['input'] for item in batch]
    outputs = [item['output'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'original_labels': original_labels,
        'instruction': instructions,
        'input': inputs,
        'output': outputs
    }

# ====================================================================
# training process recording functionality
# ====================================================================

def save_fedavg_latency_metrics(
    round_num,
    round_training_time,
    cumulative_training_time,
    filename
):
    """
    Save FedAvg latency metrics to txt file
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    lines = [
        f"FedAvg Round: {round_num}",
        f"Round Training Time: {round_training_time:.3f} ms",
        f"Cumulative Training Time: {cumulative_training_time:.3f} ms",
        "-" * 50
    ]
    
    with open(filename, 'a', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

def save_fedavg_aggregation_metrics(
    fed_round,
    epoch,
    agg_step,
    total_aggregations,
    client_losses,
    communication_volume,
    cumulative_communication_volume,
    filename
):
    """Save metrics for each aggregation step"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Calculate average loss
    avg_loss = sum(client_losses) / len(client_losses) if client_losses else 0.0
    
    # Convert communication volume to bytes
    communication_bytes = int(communication_volume * 1024 * 1024)  # Convert MB to bytes
    cumulative_communication_bytes = int(cumulative_communication_volume * 1024 * 1024)
    
    lines = [
        f"FedRound: {fed_round}",
        f"Epoch: {epoch}",
        f"Aggregation Step: {agg_step}/{total_aggregations}",
        f"Communication Volume (Bytes): {communication_bytes}",
        f"Communication Volume (MB): {communication_volume:.4f}",
        f"Cumulative Communication Volume (Bytes): {cumulative_communication_bytes}",
        f"Cumulative Communication Volume (MB): {cumulative_communication_volume:.4f}",
    ]
    
    # Add losses for each client
    for i, loss in enumerate(client_losses):
        lines.append(f"Client {i+1} Loss: {loss:.6f}")
    
    lines.extend([
        f"Average Loss: {avg_loss:.6f}",
        "-" * 50
    ])
    
    with open(filename, 'a', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

def save_fedavg_metrics(
    round_num, 
    communication_volume, 
    cumulative_communication_volume, 
    eval_results, 
    best_loss, 
    filename
):
    """Save FedAvg federated learning metrics, format completely consistent with multi-GPU mode"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
            # Get multi-GPU style data
    client_valid_losses = eval_results.get('client_valid_losses', [])
    client_restored_mses = eval_results.get('client_restored_mses', [])
    avg_valid_loss = eval_results.get('avg_valid_loss', 0.0)
    
            # Convert communication volume to bytes (consistent with multi-GPU format)
    communication_bytes = int(communication_volume * 1024 * 1024)  # Convert MB to bytes
    cumulative_communication_bytes = int(cumulative_communication_volume * 1024 * 1024)
    
    lines = [
        f"Round: {round_num}",
        f"Communication Volume (Bytes): {communication_bytes}",
        f"Communication Volume (MB): {communication_volume:.4f}",
        f"Cumulative Communication Volume (Bytes): {cumulative_communication_bytes}",
        f"Cumulative Communication Volume (MB): {cumulative_communication_volume:.4f}",
    ]
    
            # Add validation loss and restored MSE for each client (completely consistent with multi-GPU format)
    for i, (valid_loss, restored_mse) in enumerate(zip(client_valid_losses, client_restored_mses)):
        lines.append(f"Client {i} Validation Loss: {valid_loss:.4f}")
        lines.append(f"Client {i} Restored MSE: {restored_mse:.6f}")
    
    lines.extend([
        f"Average Validation Loss: {avg_valid_loss:.4f}",
        f"Best Average Validation Loss: {best_loss:.4f}",
        "-" * 50
    ])
    
    with open(filename, 'a', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

def calculate_fedavg_communication_volume(client_models):
    """Calculate FedAvg communication overhead (LoRA parameters)"""
    total_bytes = 0
    for model in client_models:
        if hasattr(model, 'peft_config'):  # PEFT model
            # Calculate LoRA parameter size
            for name, param in model.named_parameters():
                if 'lora_A' in name or 'lora_B' in name:
                    total_bytes += param.numel() * param.element_size()
    
    mb_size = total_bytes / (1024 * 1024)  # Convert to MB
    return mb_size

def print_fedavg_trainable_parameters(model, log_file=None):
    """Print FedAvg model trainable parameter information - includes complete component statistics"""
    if hasattr(model, 'print_trainable_parameters'):
        # PEFT model - first show default statistics
        model.print_trainable_parameters()
        
        # Then show detailed component statistics
        trainable_params = 0
        all_param = 0
        lora_params = {"A": 0, "B": 0}
        regression_head_params = 0
        num_encoder_params = 0
        frozen_params = 0
        
        for name, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                if 'lora_A' in name:
                    lora_params["A"] += param.numel()
                elif 'lora_B' in name:
                    lora_params["B"] += param.numel()
                elif 'regression_head' in name:
                    regression_head_params += param.numel()
                elif 'num_encoder' in name:
                    num_encoder_params += param.numel()
            else:
                frozen_params += param.numel()
        
        info = (
            f"\n[FedAvg] Detailed Parameter Statistics\n"
            f"{'='*60}\n"
            f"Total Parameters: {all_param:,}\n"
            f"Trainable Parameters: {trainable_params:,}\n"
            f"Frozen Parameters: {frozen_params:,}\n"
            f"Trainable Ratio: {100 * trainable_params / all_param:.2f}%\n"
            f"\nComponent Detailed Statistics:\n"
            f"  LoRA_A: {lora_params['A']:,}\n"
            f"  LoRA_B: {lora_params['B']:,}\n"
            f"  LoRA Total: {lora_params['A'] + lora_params['B']:,}\n"
            f"  Regression Head: {regression_head_params:,}\n"
            f"  Numeric Encoder: {num_encoder_params:,}\n"
            f"\nTrainable Component Details:\n"
            f"  LoRA Parameters: {lora_params['A'] + lora_params['B']:,}\n"
            f"  Other Components: {regression_head_params + num_encoder_params:,}\n"
            f"{'='*60}\n"
        )
        
        print(info)
        
        # Save to file (if file path is specified)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(info + "\n")
    else:
        print("Model does not have print_trainable_parameters method")




''' Class definition for recording losses '''
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Set random seed here
set_seed(42)

 




class GPT2ForRegression(GPT2PreTrainedModel):
    def __init__(self, config):
        super(GPT2ForRegression, self).__init__(config)
        self.transformer = GPT2Model(config)
        
        # Regression head: directly predict 24 time steps from text representation
        self.regression_head = nn.Linear(config.n_embd, 24)  # Output 24 dimensions
        
        # Initialize regression head weights to be more suitable for large value ranges
        # Use smaller initialization values to avoid initial predictions being too large
        torch.nn.init.xavier_uniform_(self.regression_head.weight, gain=0.1)
        torch.nn.init.zeros_(self.regression_head.bias)
        
        self.loss_fct = nn.MSELoss()
        self.model_parallel = False
        self.is_parallelizable = False

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Text part
        last_hidden = self.transformer(input_ids, attention_mask=attention_mask)[0]
        text_repr = last_hidden.mean(dim=1)  # (batch, n_embd)

        # Regression head
        logits = self.regression_head(text_repr)  # (batch, 24)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.float(), labels.float())
        
        return {"loss": loss, "logits": logits}


        


class FederatedLoRATrainer:
    def __init__(
        self,
        config: FederatedConfig = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        multi_gpu: bool = True  # New: whether to use multi-GPU training
    ):
        # Use provided config or create default
        self.config = config if config is not None else FederatedConfig()
        
        # Extract parameters from config
        self.model_name = self.config.model_name
        self.num_clients = self.config.num_clients
        self.lora_rank = self.config.lora_rank
        self.lora_alpha = self.config.lora_alpha
        self.lora_dropout = self.config.lora_dropout
        self.selective_layers = self.config.selective_layers
        self.device = device
        self.multi_gpu = multi_gpu
        
        # Set global device variable for prediction
        global global_device
        global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Prioritize using local model path (customize for your setup)
        local_model_path = os.path.join(self.config.base_data_dir, "models", "gpt2_offline")
        
        # Initialize tokenizer - prioritize local model
        try:
            if os.path.exists(local_model_path):
                print(f"Using local model to initialize tokenizer: {local_model_path}")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    local_model_path, 
                    local_files_only=True
                )
            else:
                # Try loading from cache
                print(f"Trying to load tokenizer from cache: {self.model_name}")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    self.model_name, 
                    local_files_only=True
                )
        except Exception as e:
            print(f"Failed to load tokenizer: {e}")
            print("Creating tokenizer with default GPT2 configuration...")
            from transformers import GPT2Config
            config = GPT2Config()
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token # Padding token, using eos instead of 0, eos has id 50256 in GPT2 vocabulary
        self.eval_loss = AverageMeter()  # For recording validation loss
        self.training_loss = AverageMeter()  # For recording training loss

        # Get base model to determine number of layers - prioritize local model
        try:
            if os.path.exists(local_model_path):
                print(f"Using local model to determine layer count: {local_model_path}")
                base_model = GPT2Model.from_pretrained(
                    local_model_path,
                    local_files_only=True,
                    attn_implementation="eager"
                )
            else:
                # Try loading from cache
                print(f"Trying to load model from cache to determine layer count: {self.model_name}")
                base_model = GPT2Model.from_pretrained(
                    self.model_name,
                    local_files_only=True,
                    attn_implementation="eager"
                )
        except Exception as e:
            print(f"Model loading failed: {e}")
            print("Using default GPT2 configuration...")
            from transformers import GPT2Config
            config = GPT2Config()
            base_model = GPT2Model(config)
        
        # Get number of transformer layers
        self.num_layers = len(base_model.h)  # GPT2Model.transformer.h → GPT2Model.h
        del base_model# Free memory

        # Determine target layers based on selection
        # We will only apply LoRA to the final N layers
        """Get the specific layers where LoRA will be applied"""
        self.selective_layers = min(self.selective_layers, self.num_layers)
        target_modules = self._get_target_modules()

        # LoRA configuration that will only target specified layers,应用LoRA的配置
        self.lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, 
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=target_modules,
            bias="none",
            modules_to_save=None,
        )

        print(f"Model has {self.num_layers} layers, applying LoRA to final {self.selective_layers} layers")
        print(f"Target modules: {target_modules}")
        
        # Multi-GPU allocation: assign clients to different GPUs to improve parallel efficiency
        self.client_devices = []
        if torch.cuda.is_available():
            print(f"\n🚀 Multi-GPU mode: detected {torch.cuda.device_count()} GPUs, assigning clients to different GPUs")
            # Multi-GPU allocation strategy: distribute clients across GPUs
            for i in range(self.num_clients):
                if torch.cuda.is_available() and i == 0:
                    client_device = torch.device(f"cuda:{0}")
                elif torch.cuda.is_available() and i == 1:
                    client_device = torch.device(f"cuda:{0}")
                elif torch.cuda.is_available() and i == 2:
                    client_device = torch.device(f"cuda:{0}")
                elif torch.cuda.is_available() and i == 3:
                    client_device = torch.device(f"cuda:{0}")
                elif torch.cuda.is_available() and i == 4:
                    client_device = torch.device(f"cuda:{0}")
                else:
                    client_device = torch.device("cpu")
                self.client_devices.append(client_device)
                print(f"  client {i} -> device {client_device} (Multi-GPU mode: distributed devices)")
        else:
            # CPU mode
            print(f"\n📱 CPU mode: CUDA not available")
            for i in range(self.num_clients):
                client_device = torch.device("cpu")
                self.client_devices.append(client_device)
                print(f"  client {i} -> device {client_device}")

        # Initialize client models，clientmodel
        self.client_models = self._initialize_client_models()

        # Global model 
        self.global_model = self._initialize_global_model()

    def _get_lora_traffic(self, model: PeftModel) -> int:
        """Calculate the size of LoRA parameters in bytes."""
        total_bytes = 0
        for name, param in model.state_dict().items():
            if 'lora_A' in name or 'lora_B' in name:
                total_bytes += param.numel() * param.element_size()
        return total_bytes


    def _get_target_modules(self) -> List[str]:
        """Generate pattern for targeting only the final N transformer layers."""
        target_layers = []

        # Calculate which layers to apply LoRA to (only the last N layers)
        start_layer = self.num_layers - self.selective_layers

        """In GPT-2, transformer layers are zero-indexed (starting from layer 0)"""
        for i in range(start_layer, self.num_layers):
            target_layers.append(f"transformer.h.{i}.attn.c_attn")
            # Optionally, you could add other modules like:
            # target_layers.append(f"transformer.h.{i}.attn.c_proj")
            # target_layers.append(f"transformer.h.{i}.mlp.c_fc")
            # target_layers.append(f"transformer.h.{i}.mlp.c_proj")

        return target_layers

    def _initialize_client_models(self) -> List[PeftModel]:
        """Initialize LoRA models for each client, consistent with global model."""
        client_models = []
        def hook_fn(m, i, o):
            print(f"[LoRA Used] {m.__class__.__name__} | Output has grad_fn: {hasattr(o, 'grad_fn')}")
        
        # Prioritize using local model path
        local_model_path = "/root/autodl-tmp/small/gpt2_offline"
        
        for i in range(self.num_clients):
            # Multi-GPU mode: load pretraining weights, consistent with global model
            try:
                if os.path.exists(local_model_path):
                    # Use local model
                    print(f"client {i}: Loading pretraining weights from local path: {local_model_path}")
                    gpt2_base = GPT2Model.from_pretrained(
                        local_model_path,
                        local_files_only=True,
                        attn_implementation="eager"
                    )
                else:
                    # Try loading from cache
                    print(f"client {i}: Loading pretraining weights from cache: {self.model_name}")
                    gpt2_base = GPT2Model.from_pretrained(
                        self.model_name,
                        local_files_only=True,
                        attn_implementation="eager"
                    )
            except Exception as e:
                print(f"client {i}: Pretraining weight loading failed, using default config: {e}")
                gpt2_base = GPT2Model(GPT2Config())
            
            # Initialize regression model with pretraining config
            base_model = GPT2ForRegression(gpt2_base.config)
            # Load pretraining weights for transformer part
            base_model.transformer.load_state_dict(gpt2_base.state_dict())
            
            # Apply LoRA adapter
            lora_model = get_peft_model(base_model, self.lora_config)
            
            # Set parameter gradients
            for name, param in lora_model.named_parameters():
                # Freeze only original transformer weights
                if name.startswith("transformer"):
                    param.requires_grad = False
                else:
                    # Unfreeze LoRA A/B, NumericMLPEncoder, fusion, regression_head
                    param.requires_grad = True
            # Finally move to target device (Multi-GPU allocation)
            target_device = self.client_devices[i]
            lora_model = lora_model.to(target_device)
            
            print(f"client {i} model created and moved to device: {target_device}")
            
            # Simplified validation: only check whether model is on the correct device
            model_device = next(lora_model.parameters()).device
            print(f"  model device validation: {model_device}")
            
            lora_model.print_trainable_parameters()
            client_models.append(lora_model)
            
            # Free memory
            del gpt2_base
            
            #for name, module in lora_model.named_modules():
                #if "lora_A" in name:
                    #module.register_forward_hook(hook_fn)
        return client_models

    def _initialize_global_model(self) -> PeftModel:
        # Prefer using local model path
        local_model_path = "/root/autodl-tmp/small/gpt2_offline"
        
        # Load transformer part (GPT2Model) without head
        try:
            if os.path.exists(local_model_path):
                # Use local model
                print(f"Global model: loading pretraining weights from local path: {local_model_path}")
                gpt2_base = GPT2Model.from_pretrained(
                    local_model_path,
                    local_files_only=True,
                    attn_implementation="eager"
                )
            else:
                # Try loading from cache
                print(f"Global model: loading pretraining weights from cache: {self.model_name}")
                gpt2_base = GPT2Model.from_pretrained(
                    self.model_name,
                    local_files_only=True,
                    attn_implementation="eager"
                )
        except Exception as e:
            print(f"Global model: pretraining weight loading failed, using default config: {e}")
            gpt2_base = GPT2Model(GPT2Config())

        # Initialize regression model with its config
        regression_model = GPT2ForRegression(gpt2_base.config)
        regression_model.transformer.load_state_dict(gpt2_base.state_dict())  # Load transformer parameters

        # Apply LoRA
        global_model = get_peft_model(regression_model, self.lora_config)
        for name, param in global_model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

        # Free memory
        del gpt2_base

        return global_model

    def _get_lora_state_dict(self, model: PeftModel) -> Dict[str, torch.Tensor]:
        """Extract only LoRA-A parameters from a model."""
        """Extract only LoRA A matrices from the model (exclude B matrices)"""
        # Only get keys that contain 'lora_A'
        lora_state_dict = {}  # Store extracted LoRA-A parameters
        for key, value in model.state_dict().items():
            if 'lora_A' in key:
                # Check if this belongs to one of our target layers
                is_target = False
                for target_layer in self._get_target_modules():
                    # Convert target_layer format to what appears in state_dict
                    # e.g., 'transformer.h.9.attn.c_attn' -> 'base_model.model.transformer.h.9.attn.c_attn'
                    target_in_state = f"base_model.model.{target_layer}"
                    if target_in_state in key:
                        is_target = True
                        break

                if is_target:
                    lora_state_dict[key] = value.clone()

        return lora_state_dict

    def aggregate_models(self) -> None:
        """Aggregate LoRA parameters from multi-GPU clients - multi-GPU aggregation"""
        global_lora_dict = {}

        # Get LoRA state dicts from all clients
        client_lora_dicts = []
        for i, model in enumerate(self.client_models):
            lora_dict = self._get_lora_state_dict(model)
            client_lora_dicts.append(lora_dict)
        
        # Multi-GPU: use the first client's parameter device as aggregation device
        if len(client_lora_dicts) > 0 and len(client_lora_dicts[0]) > 0:
            # Multi-GPU aggregation: dynamically choose the first client's parameter device
            aggregation_device = next(iter(client_lora_dicts[0].values())).device
        else:
            # Fallback: use the first available GPU or CPU
            aggregation_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        
        print(f"Aggregating parameters on device {aggregation_device}")

        # Move all clients' LoRA parameters to the aggregation device
        client_lora_dicts_on_device = []
        for i, lora_dict in enumerate(client_lora_dicts):
            lora_dict_on_device = {}
            for key, value in lora_dict.items():
                lora_dict_on_device[key] = value.to(aggregation_device)
            client_lora_dicts_on_device.append(lora_dict_on_device)
            print(f"Client {i} LoRA parameters moved to aggregation device {aggregation_device}")

        # Ensure all clients have the same keys (only lora_A)
        if len(client_lora_dicts_on_device) > 0:
            for key in client_lora_dicts_on_device[0].keys():
                # Aggregate only A matrices
                if 'lora_A' in key:  
                    global_lora_dict[key] = torch.stack([client_dict[key] for client_dict in client_lora_dicts_on_device]).mean(dim=0)

        print(f"Aggregation completed, processed {len(global_lora_dict)} LoRA parameters")

        # Update global model (if needed)
        if hasattr(self, 'global_model') and self.global_model is not None:
            self.global_model.to(aggregation_device)
            global_state = self.global_model.state_dict()
            global_state.update(global_lora_dict)
            self.global_model.load_state_dict(global_state, strict=False)

        # Distribute updated global parameters to all clients
        for i, client_model in enumerate(self.client_models):
            client_device = self.client_devices[i]
            
            # Move aggregated parameters to client device
            client_lora_dict = {}
            for key, value in global_lora_dict.items():
                client_lora_dict[key] = value.to(client_device)
            
            client_state = client_model.state_dict()
            client_state.update(client_lora_dict)
            client_model.load_state_dict(client_state, strict=False)
            
            print(f"Aggregated parameters distributed to client {i} on device {client_device}")

    def _temporary_aggregate_client_model(self, client_idx: int):
        pass

    def prepare_dataset(self, data, split_strategy: str = "shard") -> List[Dataset]:
        pass

    def preprocess_function(self, examples, max_length=1024, label_scale=1, target_device=None):
        pass






    def train_clients_multi_gpu_style(self, client_datasets: List[Dataset], client_eval_datasets: List[Dataset], training_args: TrainingArguments, fed_round: int):
        """Multi-GPU mode: efficient data processing to avoid OOM and correct federated aggregation logic"""
        output_paths = self.config.get_output_paths()
        base_log_dir = os.path.join(output_paths["base_dir"], "fedavg", "fedavg_v1")
        os.makedirs(base_log_dir, exist_ok=True)
        
        print(f"\nMulti-GPU training: efficient data processing + correct federated aggregation logic")
        
        # Client data file paths
        client_data_files = self.config.get_data_paths("train")
        
        # Label scales per client (data normalization)
        # Customize based on your data range
        client_label_scales = [self.config.label_scale] * self.num_clients
        
        # Prepare dataloaders and optimizers for each client
        client_train_loaders = []
        client_optimizers = []
        client_datasets_info = []
        
        for client_idx, client_model in enumerate(self.client_models):
            print(f"Preparing client {client_idx+1}/{self.num_clients} on device {self.client_devices[client_idx]}")
            
            # Ensure client model on the correct device
            client_device = self.client_devices[client_idx]
            client_model.to(client_device)
            
            # Use solar power dataset (memory-friendly) + manual split to avoid Subset issues
            train_dataset, valid_dataset = SolarDataset.create_train_val_split(
                data_file=client_data_files[client_idx],
                tokenizer=self.tokenizer,
                max_length=1024,
                label_scale=client_label_scales[client_idx],
                train_ratio=0.9,
                seed=42
            )
            
            # Create dataloader (following FedAvg_electric settings) + custom collate function
            train_loader = DataLoader(
                train_dataset,
                batch_size=training_args.per_device_train_batch_size,
                shuffle=True,
                num_workers=0,  # consistent with FedAvg_electric
                pin_memory=True if torch.cuda.is_available() else False,  # FedAvg_electric style
                collate_fn=solar_collate_fn  # use custom collate function for batch data
            )
            
            # Create optimizer (multi-GPU mode)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, client_model.parameters()),
                lr=training_args.learning_rate,
                weight_decay=training_args.weight_decay
            )
            
            client_train_loaders.append(train_loader)
            client_optimizers.append(optimizer)
            client_datasets_info.append({
                'train_size': len(train_dataset),
                'valid_size': len(valid_dataset),
                'label_scale': client_label_scales[client_idx]
            })
            
            print(f"  Dataset sizes: training={len(train_dataset)}, validation={len(valid_dataset)}")
            print(f"  Label scale: {client_label_scales[client_idx]}")
        
        # Set aggregation interval (aggregate every N batches per client)
        batches_per_aggregation = self.config.batches_per_aggregation
        print(f"\nAggregation strategy: aggregate every {batches_per_aggregation} batches")
        
        # Set aggregation metrics file path
        aggregation_metrics_file = os.path.join(base_log_dir, "aggregation_metrics.txt")
        cumulative_communication = 0.0
        print(f"Aggregation metrics will be saved to: {aggregation_metrics_file}")
        
        # Federated learning training loop
        for epoch in range(int(training_args.num_train_epochs)):
            print(f"\n----- Epoch {epoch+1}/{int(training_args.num_train_epochs)} -----")
            
            # Create iterators for each client
            client_iterators = [iter(loader) for loader in client_train_loaders]
            client_batch_counts = [0] * self.num_clients
            client_epoch_losses = [0.0] * self.num_clients
            client_epoch_batches = [0] * self.num_clients
            
            # Calculate total aggregations per epoch
            min_batches = min(len(loader) for loader in client_train_loaders)
            total_aggregations = min_batches // batches_per_aggregation
            
            print(f"Epoch {epoch+1}: each client has up to {min_batches} batches, will perform {total_aggregations} aggregations")
            
            # Train within each aggregation period
            for agg_step in range(total_aggregations):
                print(f"\n--- Aggregation step {agg_step+1}/{total_aggregations} ---")
                
                # All clients train a specified number of batches
                for client_idx in range(self.num_clients):
                    client_model = self.client_models[client_idx]
                    optimizer = client_optimizers[client_idx]
                    iterator = client_iterators[client_idx]
                    
                    client_model.train()
                    
                    # Train specified number of batches
                    for batch_in_agg in range(batches_per_aggregation):
                        try:
                            batch = next(iterator)
                        except StopIteration:
                            # Restart when data is exhausted
                            iterator = iter(client_train_loaders[client_idx])
                            batch = next(iterator)
                            client_iterators[client_idx] = iterator
                        
                        # Multi-GPU mode: directly use dataset return format, no data_collator needed
                        input_ids = batch['input_ids'].to(client_model.device)
                        attention_mask = batch['attention_mask'].to(client_model.device)
                        labels = batch['labels'].to(client_model.device)
                        
                        optimizer.zero_grad()
                        
                        
                        outputs = client_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs['loss']
                        
                        loss.backward()
                        optimizer.step()
                        
                        client_epoch_losses[client_idx] += loss.item()
                        client_epoch_batches[client_idx] += 1
                        client_batch_counts[client_idx] += 1
                
                # After all clients finish training, perform aggregation
                print(f"  All clients completed {batches_per_aggregation} training batches, start aggregation...")
                self.aggregate_models()
                print(f"  aggregation completed")
                
                # Compute communication overhead for current aggregation (uplink + downlink)
                round_communication_mb = calculate_fedavg_communication_volume(self.client_models) * 2
                cumulative_communication += round_communication_mb
                
                # Compute average loss for each client in current aggregation step
                current_client_losses = []
                for client_idx in range(self.num_clients):
                    if client_epoch_batches[client_idx] > 0:
                        avg_loss = client_epoch_losses[client_idx] / client_epoch_batches[client_idx]
                        current_client_losses.append(avg_loss)
                        print(f"  Client {client_idx+1} current average Loss: {avg_loss:.6f}")
                    else:
                        current_client_losses.append(0.0)
                
                # Save aggregation metrics
                save_fedavg_aggregation_metrics(
                    fed_round=fed_round,
                    epoch=epoch+1,
                    agg_step=agg_step+1,
                    total_aggregations=total_aggregations,
                    client_losses=current_client_losses,
                    communication_volume=round_communication_mb,
                    cumulative_communication_volume=cumulative_communication,
                    filename=aggregation_metrics_file
                )
                print(f"  Aggregation metrics saved to: {aggregation_metrics_file}")
            
            # Print final loss of each epoch
            print(f"\nEpoch {epoch+1} completed, average loss per client:")
            for client_idx in range(self.num_clients):
                if client_epoch_batches[client_idx] > 0:
                    avg_epoch_loss = client_epoch_losses[client_idx] / client_epoch_batches[client_idx]
                    print(f"  Client {client_idx+1}: {avg_epoch_loss:.6f}")
            
            # Clear GPU cache (multi-GPU mode)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Summary after training completed
        total_aggregations_in_round = sum(
            min(len(SolarDataset.create_train_val_split(
                f"/root/autodl-tmp/small/data2/train{i+1}.json",
                self.tokenizer, max_length=1024, label_scale=0.0001, train_ratio=0.9, seed=42
            )[0]) // training_args.per_device_train_batch_size for i in range(self.num_clients)) // batches_per_aggregation
            for _ in range(int(training_args.num_train_epochs))
        )
        print(f"\n=== Federated round {fed_round} completed ===")
        print(f"Total aggregations: {total_aggregations_in_round}")
        print(f"Total communication overhead: {cumulative_communication:.4f} MB")
        print(f"Aggregation metrics file: {aggregation_metrics_file}")



    def evaluate_global_model(self, client_eval_datasets: List[Dataset], batch_size: int = 4) -> Dict[str, float]:
        """Multi-GPU mode: efficient evaluation to avoid OOM and compute restored MSE"""
        
        # Client data file paths and label scales
        client_data_files = self.config.get_data_paths("train")
        client_label_scales = [self.config.label_scale] * self.num_clients
        
        eval_results = {}
        client_valid_losses = []
        client_restored_mses = []
        
        # Evaluate each client model
        for client_idx, client_model in enumerate(self.client_models):
            print(f"Evaluating Client {client_idx + 1}/{self.num_clients} Model on device {self.client_devices[client_idx]}")
                
            # Ensure client model on the correct device
            client_device = self.client_devices[client_idx]
            client_model.to(client_device)
            
            # Use solar power dataset (evaluation data) - manual split to avoid Subset issues
            _, valid_dataset = SolarDataset.create_train_val_split(
                data_file=client_data_files[client_idx],
                tokenizer=self.tokenizer,
                max_length=1024,
                label_scale=client_label_scales[client_idx],
                train_ratio=0.9,
                seed=42
            )
            
            # Create DataLoader (following FedAvg_electric settings) + custom collate function
            data_loader = DataLoader(
                valid_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False,
                collate_fn=solar_collate_fn  # use custom collate function for batch data
            )

            all_predictions_client = []
            all_labels_client = []
            total_loss_client = 0.0
            num_batches_client = 0

            client_model.eval()

            with torch.no_grad():
                for batch in data_loader:
                    input_ids = batch['input_ids'].to(client_device)
                    attention_mask = batch['attention_mask'].to(client_device)
                    labels = batch['labels'].to(client_device)
                    
                    outputs = client_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    predictions = outputs["logits"]
                    loss = outputs["loss"]

                    all_predictions_client.append(predictions.cpu())
                    all_labels_client.append(labels.cpu())
                    total_loss_client += loss.item()
                    num_batches_client += 1

            all_predictions_client = torch.cat(all_predictions_client, dim=0)
            all_labels_client = torch.cat(all_labels_client, dim=0)

            # Compute restored MSE (multi-GPU mode)
            label_scale = client_label_scales[client_idx]
            restored_predictions = all_predictions_client / label_scale
            restored_labels = all_labels_client / label_scale
            restored_mse = torch.mean((restored_predictions - restored_labels) ** 2).item()

            print(f"\nClient {client_idx} Predictions vs Ground Truth (restored to original scale):")
            for i in range(min(3, len(all_predictions_client))):
                print(f"Sample {i + 1}:")
                pred_values = all_predictions_client[i].tolist()
                true_values = all_labels_client[i].tolist()
                # Restore to original scale for display
                pred_values_rounded = [round(val / label_scale, 4) for val in pred_values]
                true_values_rounded = [round(val / label_scale, 4) for val in true_values]
                print(f"  Predicted: {pred_values_rounded}")
                print(f"  Ground Truth: {true_values_rounded}")

            avg_loss_client = total_loss_client / num_batches_client if num_batches_client > 0 else 0.0
            perplexity = torch.exp(torch.tensor(avg_loss_client)).item() if avg_loss_client > 0 else float('inf')

            # Store results (maintain original format compatibility)
            eval_results_client = {
                "loss": avg_loss_client,
                "perplexity": perplexity,
                "restored_mse": restored_mse  
            }
            eval_results[f"client_{client_idx}"] = eval_results_client  
            
            # Store for multi-GPU mode metrics saving
            client_valid_losses.append(avg_loss_client)
            client_restored_mses.append(restored_mse)
            
            print(f"Client {client_idx} Evaluation Results (on {client_device}):")
            print(f"  Loss: {avg_loss_client:.4f}")
            print(f"  Perplexity: {perplexity:.4f}")
            print(f"  Restored MSE: {restored_mse:.6f}")

        # Add summary info (for multi-GPU mode metrics)
        eval_results['client_valid_losses'] = client_valid_losses
        eval_results['client_restored_mses'] = client_restored_mses
        eval_results['avg_valid_loss'] = np.mean(client_valid_losses)

        return eval_results








    # Save global model checkpoints to specified path
    def save_global_model(self, output_dir: str):
        pass
    
    def save_client_models(self, output_dir: str):
        pass

    def predict_with_trained_models(self, data_files, batch_size=4, show_samples=5):
        """
        Use trained client models directly for prediction
        """
        print("\n🚀 Starting prediction with trained models...")
        
        # Label scales per client (consistent with training)
        client_label_scales = [self.config.label_scale] * self.num_clients
        
        # Global metrics accumulators (for generating summary txt)
        output_paths = self.config.get_output_paths()
        summary_dir = os.path.join(output_paths["plots"], "ulora_new_v1")
        os.makedirs(summary_dir, exist_ok=True)
        global_sse = 0.0  # sum of squared errors
        global_sae = 0.0  # sum of absolute errors
        global_count = 0  # number of elements
        global_mape_sum = 0.0  # sum of |err/true|
        global_mape_count = 0  # number of valid elements for MAPE
        
        for idx, (data_file, client_model, label_scale) in enumerate(zip(data_files, self.client_models, client_label_scales)):
            print(f"\n{'='*80}")
            print(f"Client {idx + 1} prediction results (data file: {data_file})")
            print(f"Label Scale: {label_scale}")
            print(f"{'='*80}")
            
            # Create dataset (consistent with training, using max_length=1024)
            dataset = SolarDataset(data_file, self.tokenizer, max_length=1024, label_scale=label_scale)
            
            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            # Prediction
            client_model.eval()
            client_model.to(global_device)
            
            all_predictions = []
            all_labels = []
            all_original_labels = []
            all_instructions = []
            all_inputs = []
            all_outputs = []
            total_loss = 0.0
            num_batches = 0
            
            with torch.no_grad():
                for batch in dataloader:
                    # Move to device
                    input_ids = batch['input_ids'].to(global_device)
                    attention_mask = batch['attention_mask'].to(global_device)
                    labels = batch['labels'].to(global_device)
                    original_labels = batch['original_labels']
                    instructions = batch['instruction']
                    inputs = batch['input']
                    raw_outputs = batch['output']
                    
                    
                    outputs = client_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    predictions = outputs['logits']
                    loss = outputs['loss']
                    
                    # Restore prediction values to original scale
                    restored_predictions = predictions.cpu() / label_scale
                    
                    all_predictions.append(restored_predictions)
                    all_labels.append(labels.cpu() / label_scale)
                    all_original_labels.append(original_labels)
                    all_instructions.extend(instructions)
                    all_inputs.extend(inputs)
                    all_outputs.extend(raw_outputs)
                    total_loss += loss.item()
                    num_batches += 1
            
            # Concatenate all results
            all_predictions = torch.cat(all_predictions, dim=0)  # (N, 24)
            all_labels = torch.cat(all_labels, dim=0)  # (N, 24)
            all_original_labels = torch.cat(all_original_labels, dim=0)  # (N, 24)
            
            # Compute metrics
            avg_loss = total_loss / num_batches
            mse = torch.mean((all_predictions - all_labels) ** 2).item()
            rmse = np.sqrt(mse)
            mae = torch.mean(torch.abs(all_predictions - all_labels)).item()
            
            # Compute MAPE (Mean Absolute Percentage Error)
            mape_values = []
            for i in range(len(all_predictions)):
                pred = all_predictions[i].numpy()
                true = all_labels[i].numpy()
                # Avoid division by zero, skip when true value is 0
                valid_indices = np.abs(true) > 1e-8
                if np.any(valid_indices):
                    mape = np.mean(np.abs((pred[valid_indices] - true[valid_indices]) / true[valid_indices])) * 100
                    mape_values.append(mape)
            
            mape = np.mean(mape_values) if mape_values else 0.0
            
            print(f"\n📊 Overall evaluation metrics:")
            print(f"  Average loss (scaled): {avg_loss:.6f}")
            print(f"  MSE (restored): {mse:.6f}")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  Total samples: {len(all_predictions)}")
            
            # ===== Global metrics accumulation =====
            diff = all_predictions - all_labels
            global_sse += torch.sum(diff ** 2).item()
            global_sae += torch.sum(torch.abs(diff)).item()
            global_count += diff.numel()
            valid_mask = torch.abs(all_labels) > 1e-8
            if torch.any(valid_mask):
                global_mape_sum += torch.sum(torch.abs(diff[valid_mask] / all_labels[valid_mask])).item()
                global_mape_count += int(valid_mask.sum().item())
            # ==========================================

            # Create directory for saving plots
            plot_dir = f"plots_ulora_new_v1/client_{idx + 1}"
            os.makedirs(plot_dir, exist_ok=True)
            
            # Generate txt file with detailed info for each sample
            txt_filename = f"client_{idx + 1}_predictions.txt"
            txt_path = os.path.join(plot_dir, txt_filename)
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Client {idx + 1} prediction results detailed report\n")
                f.write("=" * 80 + "\n")
                f.write(f"Data file: {data_file}\n")
                f.write(f"Label Scale: {label_scale}\n")
                f.write(f"Total samples: {len(all_predictions)}\n")
                f.write(f"Average loss (scaled): {avg_loss:.6f}\n")
                f.write(f"MSE (restored): {mse:.6f}\n")
                f.write(f"RMSE: {rmse:.6f}\n")
                f.write(f"MAE: {mae:.6f}\n")
                f.write(f"MAPE: {mape:.2f}%\n")
                f.write("=" * 80 + "\n\n")
                
                for i in range(len(all_predictions)):
                    f.write(f"Sample {i + 1}:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Instruction:\n{all_instructions[i]}\n\n")
                    f.write(f"Input:\n{all_inputs[i]}\n\n")
                    f.write(f"Output (ground truth):\n{all_outputs[i]}\n\n")
                    
                    # Format prediction values
                    pred_values = all_predictions[i].tolist()
                    pred_str = ",".join([f"{x:.4f}" for x in pred_values])
                    f.write(f"Prediction (predicted values):\n{pred_str}\n\n")
                    
                    # Compute sample-level metrics
                    true_values = all_original_labels[i].tolist()
                    sample_mse = np.mean((np.array(pred_values) - np.array(true_values)) ** 2)
                    sample_rmse = np.sqrt(sample_mse)
                    sample_mae = np.mean(np.abs(np.array(pred_values) - np.array(true_values)))
                    
                    # Compute sample-level MAPE
                    pred_array = np.array(pred_values)
                    true_array = np.array(true_values)
                    valid_indices = np.abs(true_array) > 1e-8
                    if np.any(valid_indices):
                        sample_mape = np.mean(np.abs((pred_array[valid_indices] - true_array[valid_indices]) / true_array[valid_indices])) * 100
                    else:
                        sample_mape = 0.0
                    
                    f.write(f"Sample metrics:\n")
                    f.write(f"  MSE: {sample_mse:.6f}\n")
                    f.write(f"  RMSE: {sample_rmse:.6f}\n")
                    f.write(f"  MAE: {sample_mae:.6f}\n")
                    f.write(f"  MAPE: {sample_mape:.2f}%\n")
                    f.write("\n" + "=" * 80 + "\n\n")
            
            print(f"📄 Detailed prediction results saved to: {txt_path}")
            
            # Show some prediction samples
            print(f"\n🔍 Prediction sample display:")
            for i in range(min(show_samples, len(all_predictions))):
                pred_values = all_predictions[i].tolist()
                true_values = all_original_labels[i].tolist()
                
                print(f"\nSample {i + 1}:")
                pred_str = [f'{x:.4f}' for x in pred_values]
                true_str = [f'{x:.4f}' for x in true_values]
                print(f"  Predicted values: {pred_str}")
                print(f"  Ground truth: {true_str}")
                
                sample_mse = np.mean((np.array(pred_values) - np.array(true_values)) ** 2)
                sample_rmse = np.sqrt(sample_mse)
                sample_mae = np.mean(np.abs(np.array(pred_values) - np.array(true_values)))
                
                # Compute sample MAPE
                pred_array = np.array(pred_values)
                true_array = np.array(true_values)
                valid_indices = np.abs(true_array) > 1e-8
                if np.any(valid_indices):
                    sample_mape = np.mean(np.abs((pred_array[valid_indices] - true_array[valid_indices]) / true_array[valid_indices])) * 100
                else:
                    sample_mape = 0.0
                
                print(f"  Sample MSE: {sample_mse:.6f}")
                print(f"  Sample RMSE: {sample_rmse:.6f}")
                print(f"  Sample MAE: {sample_mae:.6f}")
                print(f"  Sample MAPE: {sample_mape:.2f}%")
            
            # 🎨 Generate prediction result plots
            print(f"\n🎨 Generating prediction result plots...")
            
            # Create a plot for each sample
            for i in range(len(all_predictions)):
                pred_values = all_predictions[i].numpy()
                true_values = all_original_labels[i].numpy()
                
                # Create plot
                plt.figure(figsize=(12, 6))
                
                # 24 time points
                time_points = list(range(24))
                
                # Plot prediction and ground truth curves
                plt.plot(time_points, pred_values, 'b-', linewidth=2, marker='o', 
                        markersize=4, label='predict', alpha=0.8)
                plt.plot(time_points, true_values, 'r-', linewidth=2, marker='s', 
                        markersize=4, label='ground truth', alpha=0.8)
                
                # Set plot properties
                plt.title(f'client {idx + 1} - sample {i + 1} prediction vs ground truth', fontsize=14, fontweight='bold')
                plt.xlabel('time (30 minutes)', fontsize=12)
                plt.ylabel('solar power (MW)', fontsize=12)
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                
                # Set x-axis ticks
                plt.xticks(range(0, 24, 4))
                
                # Add MSE information
                sample_mse = np.mean((pred_values - true_values) ** 2)
                plt.text(0.02, 0.98, f'MSE: {sample_mse:.6f}', 
                        transform=plt.gca().transAxes, 
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                # Save plot
                plot_filename = f"client_{idx + 1}_sample_{i + 1}.png"
                plot_path = os.path.join(plot_dir, plot_filename)
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"  ✅ Sample {i + 1} plot saved to: {plot_path}")
            
            print(f"📊 Client {idx + 1} all plots saved to: {plot_dir}")
            print(f"📄 Client {idx + 1} detailed prediction results saved to: {txt_path}")
        
        # ====== Output global summary metrics to txt ======
        if global_count > 0:
            total_mse = global_sse / global_count
            total_rmse = float(np.sqrt(total_mse))
            total_mae = global_sae / global_count
            total_mape = (global_mape_sum / global_mape_count) * 100 if global_mape_count > 0 else 0.0
            summary_path = os.path.join(summary_dir, "summary_metrics.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("FedAvg direct prediction summary metrics\n")
                f.write("=" * 80 + "\n")
                f.write(f"Total samples: {global_count}\n")
                f.write(f"MSE: {total_mse:.6f}\n")
                f.write(f"RMSE: {total_rmse:.6f}\n")
                f.write(f"MAE: {total_mae:.6f}\n")
                f.write(f"MAPE: {total_mape:.2f}%\n")
            print(f"\n📊 Summary metrics saved to: {summary_path}")
        
        print(f"\n🎉 Prediction completed! All results saved to {summary_dir} directory")

    def run_federated_learning(
            self,
            num_rounds: int = 5,
            local_batch_size: int = 8,
            local_epochs: int = 30,
            learning_rate: float = 1e-4,
            split_strategy: str = "shard", # Data processing type: non-overlapping or random sampling
            output_dir: str = "./fed_lora_gpt2" # Global model checkpoints save path
    ):
        """
        Run federated learning process
        New aggregation strategy: 3 aggregations per epoch instead of 1 aggregation every 3 epochs
        """
        """Run the complete federated learning process with detailed logging."""
        output_paths = self.config.get_output_paths()
        base_dir = os.path.join(output_paths["base_dir"], "ulora_new", "ulora_v1")
        os.makedirs(base_dir, exist_ok=True)
        
        latency_file = os.path.join(base_dir, "fedavg_latency.txt")
        metrics_file = os.path.join(base_dir, "fedavg_metrics.txt")
        params_file = os.path.join(base_dir, "fedavg_trainable_parameters.txt")
        
        model_save_dir = os.path.join(base_dir, "trained_models")
        
        # Initialize cumulative time and communication overhead
        cumulative_training_time = 0.0
        cumulative_communication = 0.0
        best_loss = float('inf')
        
        # Record model parameters info
        print("\n=== FedAvg model parameters info ===")
        print_fedavg_trainable_parameters(self.global_model, params_file)
        
        print(f"\n=== Multi-GPU mode: skip preloading datasets, use lazy loading ===")
        print("Data will be loaded directly from files during training to avoid memory usage")
        print(f"\n=== New aggregation strategy: aggregate every 55 batches ===")
        print("All clients train 55 batches simultaneously, then aggregate and distribute parameters to all clients")
        
        # Create empty placeholder lists (maintain interface compatibility)
        client_datasets = [None] * self.num_clients  # Placeholder
        client_eval_datasets = [None] * self.num_clients  # Placeholder

        # Define training arguments, this is the hyperparameters section for training
        training_args = TrainingArguments(
            output_dir=os.path.join(base_dir, "checkpoints"),# This parameter is used with save_strategy, model checkpoints save path
            num_train_epochs=local_epochs, # This specifies how many epochs each client trains in each federated round
            per_device_train_batch_size=local_batch_size,
            learning_rate=learning_rate,# Learning rate
            weight_decay=0.01, # AdamW optimizer weight decay parameters
            logging_dir=os.path.join(base_dir, "logs"),# Training logs will be saved to logging_dir directory
            logging_steps=10, # Record logs every N training batches, including loss values
            save_strategy="epoch",  # Don't save checkpoints during training
            logging_strategy="epoch",  # Log by epoch
            #evaluation_strategy="epoch",  # Evaluate every epoch
            report_to="none",  # Disable wandb, tensorboard, etc.
            remove_unused_columns=False,  # Keep all columns in the dataset
            label_names=["labels"]
        )
        total_comm = 0
        # Run federated learning rounds with detailed logging
        for round_idx in range(num_rounds):
            print(f"\n----- Federated Learning Round {round_idx + 1}/{num_rounds} -----")
            
            # Record round start time
            round_start_time = time.time()
            
            # Train each client on their local data
            self.train_clients_multi_gpu_style(client_datasets, client_eval_datasets, training_args, fed_round = round_idx + 1)
            
            # Record round end time and calculate training time
            round_end_time = time.time()
            round_training_time = (round_end_time - round_start_time) * 1000  # Convert to milliseconds
            cumulative_training_time += round_training_time
            
            # Save latency metrics
            save_fedavg_latency_metrics(
                round_idx + 1, 
                round_training_time, 
                cumulative_training_time, 
                latency_file
            )
            
            # Evaluate each client model on its local validation set
            eval_results = self.evaluate_global_model(client_eval_datasets)
            print(f"Round {round_idx + 1} Evaluation: {eval_results}")
            
            total_epochs = int(training_args.num_train_epochs)
            batches_per_aggregation = 55  # Consistent with training loop settings
            
            # Estimate aggregations per epoch (based on minimum dataset size)
            min_batches_per_epoch = min(len(SolarDataset.create_train_val_split(
                f"/root/autodl-tmp/small/data2/train{i+1}.json",
                self.tokenizer, max_length=1024, label_scale=0.0001, train_ratio=0.9, seed=42
            )[0]) // training_args.per_device_train_batch_size for i in range(self.num_clients))
            
            aggregations_per_epoch = min_batches_per_epoch // batches_per_aggregation
            total_aggregations = total_epochs * aggregations_per_epoch
            
            print(f"  Estimated communication overhead: {total_epochs} epochs × {aggregations_per_epoch} aggregations/epoch = {total_aggregations} total aggregations")
            
            # Calculate total communication overhead
            down_bytes = self._get_lora_traffic(self.global_model)  # Downlink per client
            upload_bytes = [self._get_lora_traffic(cm) for cm in self.client_models]
            total_down = down_bytes * self.num_clients * total_aggregations  # Downlink to all clients
            total_up   = sum(upload_bytes) * total_aggregations         # Sum of client uploads
            total_comm_per_round = total_down + total_up
            total_comm += total_comm_per_round
            
            # Also calculate communication overhead based on our new function (for comparison)
            round_communication_mb = calculate_fedavg_communication_volume(self.client_models) * 2 * total_aggregations  # Uplink + downlink
            cumulative_communication += round_communication_mb
            
            # Calculate best loss
            avg_loss = eval_results.get('avg_valid_loss', 0.0)  # Use new average validation loss
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            # Save federated learning metrics
            save_fedavg_metrics(
                round_idx + 1,
                round_communication_mb,  # Now each round has communication overhead
                cumulative_communication,
                eval_results,
                best_loss,
                metrics_file
            )
            
            print(f"Round {round_idx + 1} Training Time: {round_training_time:.3f} ms")
            print(f"Round {round_idx + 1} Communication: {round_communication_mb:.6f} MB")
            print(f"Round {round_idx + 1} Average Loss: {avg_loss:.6f}")
                
        print(f"\n=== FedAvg training completed ===")
        print(f"Grand total communication overhead over {num_rounds} rounds: {total_comm/1e6:.2f} MB")
        print(f"New calculation - Total communication: {cumulative_communication:.6f} MB")
        print(f"Total training time: {cumulative_training_time:.3f} ms")
        print(f"Best loss achieved: {best_loss:.6f}")
        
        # Save final summary information
        summary_lines = [
            "=" * 60,
            "FedAvg Federated Learning Training Summary",
            "=" * 60,
            f"Training rounds: {num_rounds}",
            f"Number of clients: {len(self.client_models)}",
            f"Local training epochs per round: {local_epochs}",
            f"LoRA rank: {self.lora_rank}",
            f"Best validation loss: {best_loss:.6f}",
            f"Total training time: {cumulative_training_time:.3f} ms",
            f"Total communication overhead: {cumulative_communication:.6f} MB",
            "=" * 60
        ]
        
        with open(metrics_file, 'a', encoding='utf-8') as f:
            f.write("\n".join(summary_lines) + "\n")
        
        # After training completed, directly perform prediction without saving model
        print("\n🚀 Training completed! Starting direct prediction...")
        
        # Prediction data file paths
        predict_data_files = self.config.get_data_paths("predict")
        
        # Check if prediction data files exist
        missing_files = []
        for i, data_file in enumerate(predict_data_files):
            if not os.path.exists(data_file):
                missing_files.append(f"Client {i+1}: {data_file}")
        
        if missing_files:
            print("⚠️  Warning: The following prediction data files do not exist:")
            for missing in missing_files:
                print(f"  {missing}")
            print("Skipping prediction step, returning only training results")
            return eval_results
        
        # Perform prediction
        try:
            self.predict_with_trained_models(
                data_files=predict_data_files,
                batch_size=4,
                show_samples=5
            )
        except Exception as e:
            print(f"❌ Error occurred during prediction: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing to return training results...")
        
        return eval_results

# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("FedAvg Federated Learning Training - Multi-GPU Mode")
    print("=" * 80)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"🚀 CUDA available, detected {torch.cuda.device_count()} GPUs:")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Current CUDA version: {torch.version.cuda}")
        print(f"  PyTorch version: {torch.__version__}")
    else:
        print("⚠️  CUDA not available, will use CPU training")
    
    # Initialize configuration
    config = FederatedConfig()
    
    print(f"\n📋 Training configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Number of clients: {config.num_clients}")
    print(f"  LoRA applied layers: {config.selective_layers}")
    print(f"  Federated learning rounds: {config.num_rounds}")
    print(f"  Multi-GPU mode: {'Enabled (clients assigned to different GPUs)' if torch.cuda.is_available() else 'Disabled (no available GPUs)'}")

    # Initialize federated trainer with multi-GPU training (multi-GPU allocation mode)
    fed_trainer = FederatedLoRATrainer(
        config=config,
        multi_gpu=True  # Enable multi-GPU mode training (clients assigned to different GPUs)
    )
    
    start_time = time.time()
    print(f"\n⏰ Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    
    # Run federated learning
    results = fed_trainer.run_federated_learning(
        num_rounds=config.num_rounds,
        local_batch_size=config.local_batch_size,
        local_epochs=config.local_epochs,
        learning_rate=config.learning_rate,  
        output_dir=f"./FedAvg_{config.selective_layers}_layers"
    )
    
    end_time = time.time()
    elapsed = end_time - start_time
    mins = int(elapsed // 60)
    secs = elapsed % 60
    
    print(f"\n" + "=" * 80)
    print("🎉 FedAvg Multi-GPU Mode Training Completed - Final Statistics")
    print("=" * 80)
    print(f"⏰ End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"⏱️  Total wall clock time: {elapsed:.2f} seconds ({mins} minutes {secs:.2f} seconds)")
    print(f"📊 Final results: {results}")
    
    output_paths = config.get_output_paths()
    base_dir = os.path.join(output_paths["base_dir"], "ulora_new", "ulora_v1")
    execution_report_file = os.path.join(base_dir, "execution_report.txt")
    
    os.makedirs(base_dir, exist_ok=True)
    
    # Generate GPU usage report
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info.append(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    with open(execution_report_file, 'w', encoding='utf-8') as f:
        f.write("FedAvg Multi-GPU Mode Federated Learning Execution Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        f.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
        f.write(f"Total wall clock time: {elapsed:.2f} seconds ({mins} minutes {secs:.2f} seconds)\n")
        f.write(f"Model: {config.model_name}\n")
        f.write(f"Number of clients: {config.num_clients}\n")
        f.write(f"LoRA applied layers: {config.selective_layers}\n")
        f.write(f"Federated learning rounds: {config.num_rounds}\n")
        f.write(f"Training mode: Multi-GPU mode (clients assigned to different GPUs, aggregate every {config.batches_per_aggregation} batches)\n")
        f.write("GPU information:\n")
        for gpu in gpu_info:
            f.write(f"  {gpu}\n")
        f.write(f"Multi-GPU mode training: {'Enabled' if torch.cuda.is_available() else 'Disabled'}\n")
        f.write(f"Final results: {results}\n")
        f.write(f"Save directory: {base_dir}\n")
    
    print(f"\n📁 All file save paths (consistent with FedIT):")
    print(f"  - Base directory: {base_dir}")
    print(f"  - Execution report: {execution_report_file}")
    print(f"  - Detailed logs and result files are all saved in the base directory")
    
    if torch.cuda.is_available():
        print(f"\n🔧 Multi-GPU mode device allocation summary:")
        for i in range(config.num_clients):
            if i < torch.cuda.device_count():
                device_name = torch.cuda.get_device_name(i)
                device_str = f"cuda:{i}"
            else:
                device_name = "CPU"
                device_str = "CPU"
            print(f"  Client {i} -> {device_str} ({device_name})")
    
    print(f"\n✅ Multi-GPU mode FedAvg training task completed! (aggregate every {config.batches_per_aggregation} batches)")
