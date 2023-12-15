from torch import nn
from torch import tensor, no_grad

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, module_list : nn.ModuleList = None):
        super(TextClassificationModel, self).__init__()
        self.trained_epochs = 0
        self.initial_LR = 0
        self.embedding_dim = embed_dim
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        if module_list == None:
            self.hidden_layers = nn.ModuleList([
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            ])
        else:
            self.hidden_layers = module_list
        self.output_layer = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        for layer in self.hidden_layers:  # Iterate over linear layers
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        for hidden_layer in self.hidden_layers:
            embedded = hidden_layer(embedded)
        return self.output_layer(embedded)
    
    def predict(self, processed_text, text):
        with no_grad(): # disables gradient calculation
            tensor_text = tensor(processed_text.text_pipeline(text))
            if tensor_text.numel() == 0:
                raise ValueError("The tensor is empty. Please provide a non-empty tensor.")
            output = self(tensor_text, tensor([0]))
            return (output.argmax(1).item() + 1, output.tolist())
