import os

def load_tokens(file_path):
    tokens = {}
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=")
                tokens[key] = value
    return tokens

def set_environment_variables(tokens):
    for key, value in tokens.items():
        os.environ[key] = value

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# huggingface_token_read = os.environ.get("HUGGINGFACE_TOKEN_READ")
# huggingface_token_write = os.environ.get("HUGGINGFACE_TOKEN_WRITE")
# aws_access_key = os.environ.get("AWS_ACCESS_KEY")
# gcp_api_key = os.environ.get("GCP_API_KEY")
# azure_token = os.environ.get("AZURE_TOKEN")
