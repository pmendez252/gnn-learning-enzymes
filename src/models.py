from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import GATConv, global_mean_pool


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()

        x = self.conv2(x, edge_index)
        x = x.relu()

        x = global_mean_pool(x, batch)
        return self.classifier(x)


class GCN_three_layers(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()

        x = self.conv2(x, edge_index)
        x = x.relu()

        x = self.conv3(x, edge_index)
        x = x.relu()

        x = global_mean_pool(x, batch)
        return self.classifier(x)



class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()

        self.conv1 = GATConv(input_dim, hidden_dim, heads=1)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()

        x = self.conv2(x, edge_index)
        x = x.relu()

        x = global_mean_pool(x, batch)
        return self.classifier(x)


class GAT_multihead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()

        heads = 8
        assert hidden_dim % heads == 0
        features_per_head = hidden_dim // heads

        self.conv1 = GATConv(
            input_dim, features_per_head, heads=heads
        )
        self.conv2 = GATConv(
            hidden_dim, features_per_head, heads=heads
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()

        x = self.conv2(x, edge_index)
        x = x.relu()

        x = global_mean_pool(x, batch)
        return self.classifier(x)


class GAT_multihead_dropout(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()

        heads = 8
        assert hidden_dim % heads == 0
        features_per_head = hidden_dim // heads

        self.conv1 = GATConv(
            input_dim, features_per_head, heads=heads
        )
        self.conv2 = GATConv(
            hidden_dim, features_per_head, heads=heads
        )
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.dropout(x)

        x = global_mean_pool(x, batch)
        return self.classifier(x)