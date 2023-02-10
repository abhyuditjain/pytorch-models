from torchinfo import summary


def print_summary(model):
    batch_size = 20
    summary(
        model,
        input_size=(batch_size, 3, 32, 32),
        verbose=1,
        col_names=[
            "kernel_size",
            "input_size",
            "output_size",
            "num_params",
            "mult_adds",
            "trainable",
        ],
        row_settings=["var_names"],
    )
