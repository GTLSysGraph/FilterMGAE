from .edcoder import PreModel


def build_model(args, mode):
    num_heads = args.num_heads
    num_out_heads = args.num_out_heads
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    residual = args.residual
    attn_drop = args.attn_drop
    in_drop = args.in_drop
    norm = args.norm
    negative_slope = args.negative_slope
    encoder_type = args.encoder
    decoder_type = args.decoder
    mask_rate = args.mask_rate
    drop_edge_rate = args.drop_edge_rate
    replace_rate = args.replace_rate

    activation = args.activation
    loss_fn = args.loss_fn
    alpha_l = args.alpha_l
    concat_hidden = args.concat_hidden
    num_features = args.num_features

    # add by ssh
    mode                  = mode
    scheme                = args.scheme
    num_nodes             = args.num_nodes
    mask_strategy         = args.mask_strategy
    keep_scope_centerline = args.keep_scope_centerline
    keep_scope_interval   = args.keep_scope_interval
    mask_scope_centerline = args.mask_scope_centerline
    mask_scope_interval   = args.mask_scope_interval
    keep_num_ratio        = args.keep_num_ratio
    mask_num_ratio        = args.mask_num_ratio 

    model = PreModel(
        in_dim=num_features,
        num_hidden=num_hidden,
        num_layers=num_layers,
        nhead=num_heads,
        nhead_out=num_out_heads,
        activation=activation,
        feat_drop=in_drop,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        mask_rate=mask_rate,
        norm=norm,
        loss_fn=loss_fn,
        drop_edge_rate=drop_edge_rate,
        replace_rate=replace_rate,
        alpha_l=alpha_l,
        concat_hidden=concat_hidden,
        # add by ssh
        mode                  = mode,
        scheme                = scheme,
        num_nodes             = num_nodes,
        mask_strategy         = mask_strategy,
        keep_scope_centerline = keep_scope_centerline,
        keep_scope_interval   = keep_scope_interval,
        mask_scope_centerline = mask_scope_centerline,
        mask_scope_interval   = mask_scope_interval,
        keep_num_ratio        = keep_num_ratio,
        mask_num_ratio        = mask_num_ratio
    )
    return model
