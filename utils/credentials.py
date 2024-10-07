import openai

def api_key_list(api_group):
    if api_group == '<API_GROUP_NAME1>':
        api_key_list = [
            {
                "api_key": "<API_KEY",
                "api_version": "<API_VERSION>",
                "azure_endpoint": "<AZURE_ENDPOINT>",
                "model": "<MODEL_NAME>"
            },
        ]
    elif api_group == '<API_GROUP_NAME2>':
        api_key_list = [
            {
                "api_key": "<API_KEY",
                "api_version": "<API_VERSION>",
                "azure_endpoint": "<AZURE_ENDPOINT>",
                "model": "<MODEL_NAME>"
            },
        ]
    return api_key_list