class Configs:
    input_channels = 1
    mid_channels = 64
    final_out_channels = 128
    stride = 1
    dropout = 0.3
    trans_dim = 500
    num_heads = 5
    num_classes = 2 

class HParams:
    lambda_u = 1.0
    conf_thresh = 0.8
    total_epochs = 100
