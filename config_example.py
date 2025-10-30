# Example Configuration for Federated Learning
# Copy this to your own config file and modify as needed


class MyFederatedConfig:
    def __init__(self):
        
        
        # Customize these parameters for your use case
        self.model_name = "gpt2"
        self.num_clients = 3
        self.selective_layers = 6
        
        # LoRA configuration
        self.lora_rank = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.1
        
        # Training configuration
        self.num_rounds = 12
        self.local_epochs = 3
        self.local_batch_size = 4
        self.learning_rate = 0.0001
        
        # Data configuration
        self.max_length = 1024
        self.label_scale = 15
        self.batches_per_aggregation = 55
        
        # Paths - customize these for your setup
        self.base_data_dir = "./data"
        self.output_dir = "./results"
        
        # Data file names - customize based on your dataset
        self.train_files = ['train1.json', 'train2.json', 'train3.json']
        self.predict_files = ['predict1.json', 'predict2.json', 'predict3.json']
