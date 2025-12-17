def estimate_gcn_flops(data, model):
    flops = 0
    num_edges = data.edge_index.size(1)
    num_nodes = data.num_nodes
    layer_inputs = data.num_features

    for layer in model.convs:
        layer_outputs = layer.out_channels
        flops += 2 * num_edges * layer_inputs  # Sparse matmul: A @ X
        flops += num_nodes * layer_inputs * layer_outputs  # Linear transformation
        layer_inputs = layer_outputs

    return flops / 1e6  # FLOPs in MegaFLOPs
