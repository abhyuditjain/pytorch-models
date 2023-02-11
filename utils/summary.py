from torchinfo import summary


def print_summary(model, device, input_size, batch_size=20):
    model = model.to(device=device)
    s = summary(
        model,
        input_size=(batch_size, *input_size),
        verbose=0,
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

    print(s)
