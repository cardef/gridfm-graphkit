import torch
from torch import nn
from torch_geometric.nn import HeteroConv, TransformerConv
from gridfm_graphkit.io.registries import MODELS_REGISTRY
from gridfm_graphkit.io.param_handler import get_physics_decoder
from torch_scatter import scatter_add
from gridfm_graphkit.models.utils import (
    ComputeBranchFlow,
    ComputeNodeInjection,
    ComputeNodeResiduals,
    bound_with_sigmoid,
)
from gridfm_graphkit.datasets.globals import (
    # Bus feature indices
    VM_H,
    VA_H,
    MIN_VM_H,
    MAX_VM_H,
    # Output feature indices
    VM_OUT,
    PG_OUT_GEN,
    # Generator feature indices
    PG_H,
    MIN_PG,
    MAX_PG,
    BUS_OUT_DIMENSIONS,
    GEN_OUT_DIMENSIONS,
)





@MODELS_REGISTRY.register("FullyConnectedNN")
class FullyConnectedNN(nn.Module):
    """
    
    """

    def __init__(self, args) -> None:
        super().__init__()
        self.num_layers = args.model.num_layers
        self.hidden_dim = args.model.hidden_size
        self.input_bus_dim = args.model.input_bus_dim
        self.input_gen_dim = args.model.input_gen_dim
        self.edge_dim = args.model.edge_dim
        self.task = args.task.task_name
        self.dropout = getattr(args.model, "dropout", 0.0)

        # projections for each node type
        self.input_proj_bus = nn.Sequential(
            nn.Linear(self.input_bus_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        self.input_proj_gen = nn.Sequential(
            nn.Linear(self.input_gen_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        self.input_proj_edge = nn.Sequential(
            nn.Linear(self.edge_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        # Build hetero layers: HeteroConv of TransformerConv per relation
        self.layers = nn.ModuleList()
        self.norms = nn.LayerNorm(self.hidden_dim)
        for i in range(self.num_layers):
            layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.layers.append(layer)

            # Norms for node representations (note: after HeteroConv each node type will have size out_dim * heads)
            

        # Separate shared MLPs to produce final bus/gen outputs (predictions y)
        self.mlp_bus = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, BUS_OUT_DIMENSIONS),
        )

        self.mlp_gen = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, GEN_OUT_DIMENSIONS),
        )

        self.activation = nn.LeakyReLU()


    def forward(self, x_dict, edge_index_dict, edge_attr_dict, mask_dict):
        """
        x_dict: {"bus": Tensor[num_bus, bus_feat], "gen": Tensor[num_gen, gen_feat]}
        edge_index_dict: keys like ("bus","connects","bus"), ("gen","connected_to","bus"), ("bus","connected_to","gen")
        edge_attr_dict: same keys -> edge attributes (bus-bus requires G,B)
        batch_dict: dict mapping node types to batch tensors (if using batching). Not used heavily here but kept for API parity.
        mask: optional mask per node (applies when computing residuals)
        """


        # 1) initial projections
        h_bus = self.input_proj_bus(x_dict["bus"])  # [num_bus, hidden_dim]
        h_gen = self.input_proj_gen(x_dict["gen"])  # [num_gen, hidden_dim]

        # concatenate data for forward propagation
        combined_data = torch.cat((h_bus, h_gen), dim=0)

        num_bus = x_dict["bus"].size(0)

        # iterate layers
        for layer in self.layers:
            layer_output = layer(combined_data) # [Nb+Ng, hidden_dim]
            layer_output = self.norms(layer_output) # [Nb+Ng, hidden_dim]
            layer_output = self.activation(layer_output)

            # # skip connection
            combined_data = combined_data + layer_output
            
        # split data
        # print(f'\n\n\n\n\n{combined_data.shape=}')
        out_bus = combined_data[:num_bus]
        out_gen = combined_data[num_bus:]
        # print(f'\n\n\n\n\n{out_bus.shape=}')
        # print(f'\n\n\n\n\n{out_gen.shape=}')


        # Decode bus and generator predictions
        output_temp = self.mlp_bus(out_bus)  # [num_buses, 4] -> [Vm, Va, Pg, Qg]
        gen_temp = self.mlp_gen(out_gen)  # [num_gens, 1]  -> Pg



        # print(f'\n\n\n\n\n{output_temp.shape=}')
        # print(f'{gen_temp.shape=}')


        return {"bus": output_temp, "gen": gen_temp}