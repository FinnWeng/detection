import ml_collections

def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.name = 'ViT-B_16'
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    # config.hidden_size = 768
    config.hidden_size = 128
    config.transformer = ml_collections.ConfigDict()
    # config.transformer.mlp_dim = 3072
    config.transformer.mlp_dim = 128
    config.transformer.num_heads = 4
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.name = 'ViT-B_32'
    config.patches.size = (6, 6)
    return config


def get_swin_config():
    config = ml_collections.ConfigDict()
    config.name = 'swin_tiny_patch4_window7_224'
    config.include_top=True
    config.img_size=(224, 224)
    # config.img_size=(96, 96)
    config.patch_size=(4, 4)
    config.in_chans=3
    # config.num_classes=1000
    config.embed_dim=96
    config.depths=[2, 2, 6, 2]
    config.num_heads=[3, 6, 12, 24]
    config.window_size=7
    # config.window_size=3
    config.mlp_ratio=4.
    config.qkv_bias=True
    config.qk_scale=None
    config.drop_rate=0.
    config.attn_drop_rate=0.
    config.drop_path_rate=0.1
    # config.norm_layer=tf.keras.layers.LayerNormalization
    config.ape=False
    config.patch_norm=True
    config.use_checkpoint=False

    return config


