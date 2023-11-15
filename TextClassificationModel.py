from torch import nn
from torch import tensor, no_grad

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.trained_epochs = 0
        self.initial_LR = 0
        self.embedding_dim = embed_dim
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        ])
        self.output_layer = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        for i in range(0, len(self.hidden_layers), 2):  # Iterate over linear layers
            layer = self.hidden_layers[i]
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
            text = tensor(processed_text.text_pipeline(text))
            output = self(text, tensor([0]))
            return (output.argmax(1).item() + 1, output.tolist())
