from gridfm_graphkit.io.registries import MODELS_REGISTRY
from torch import nn
import torch

from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.models.layer import new_layer_config, MLP



class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension


    TODO replace 'register' with local version of it

    """
    def __init__(
                self, 
                dim_in,
                dim_inner,
                args
                ):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if args.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                args.node_encoder_name]
            self.node_encoder = NodeEncoder(dim_inner)
            if args.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(dim_inner, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            # Update dim_in to reflect the new dimension fo the node features
            self.dim_in = dim_inner
        if args.edge_encoder:
            # Hard-limit max edge dim for PNA.
            if 'PNA' in args.model.gt.layer_type:   # TODO remove condition if PNA not needed
                dim_edge = min(128, dim_inner)
            else:
                dim_edge = dim_inner
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(dim_edge, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


@MODELS_REGISTRY.register("GRIT")
class GritTransformer(torch.nn.Module):
    '''
        The proposed GritTransformer (Graph Inductive Bias Transformer)
    '''

    def __init__(self, args):
        super().__init__()

        # ### TODO remove default args not needed ####
        # self.input_dim = 
        # self.hidden_dim = 
        # self.output_dim = 
        # self.edge_dim = 
        # self.num_layers = args.model.num_layers
        # self.heads = getattr(args.model, "attention_head", 1)
        # self.dropout = getattr(args.model, "dropout", 0.0)
        # ### ###

        dim_in = args.model.input_dim
        dim_out = args.model.output_dim
        dim_inner = args.model.hidden_size
        dim_edge = args.model.edge_dim
        num_heads = args.model.attention_head
        dropout = args.model.dropout
        num_layers = args.model.num_layers
        
        self.encoder = FeatureEncoder(
                        dim_in, 
                        dim_inner,
                        args.model.encoder
                        )   # TODO add args
        dim_in = self.encoder.dim_in    


        if args.model.posenc_RRWP.enable:
            # TODO connect 'register' to local version
            self.rrwp_abs_encoder = register.node_encoder_dict["rrwp_linear"]\
                (args.model.posenc_RRWP.ksteps, dim_inner)
            rel_pe_dim = args.model.posenc_RRWP.ksteps
            self.rrwp_rel_encoder = register.edge_encoder_dict["rrwp_linear"] \
                (rel_pe_dim, dim_edge,
                 pad_to_full_graph=args.model.gt.attn.full_attn,
                 add_node_attr_as_self_loop=False,
                 fill_value=0.
                 )


        if args.model.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, dim_inner, args.model.layers_pre_mp)
            dim_in = dim_inner

        assert args.model.hidden_size == dim_inner == dim_in, \
            "The inner and hidden dims must match."

        global_model_type = args.model.gt.layer_type
        # global_model_type = "GritTransformer"
        # TODO replace this with local register logic
        TransformerLayer = register.layer_dict.get(global_model_type)

        layers = []
        for ll in range(num_layers):
            layers.append(TransformerLayer(
                in_dim=args.model.gt.dim_hidden,
                out_dim=args.model.gt.dim_hidden,
                num_heads=num_heads,
                dropout=dropout,
                act=args.model.act,
                attn_dropout=args.model.gt.attn_dropout,
                layer_norm=args.model.gt.layer_norm,
                batch_norm=args.model.gt.batch_norm,
                residual=True,
                norm_e=args.model.gt.attn.norm_e,
                O_e=args.model.gt.attn.O_e,
                cfg=args.model.gt,
            ))

        self.layers = nn.Sequential(*layers)

        self.decoder = nn.Sequential(
            nn.Linear(dim_inner, dim_inner),
            nn.LeakyReLU(),
            nn.Linear(dim_inner, dim_out),
        )

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)

        return batch