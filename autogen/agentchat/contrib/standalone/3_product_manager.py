import os

from standalone_assistant_agent import StandAloneAssistantAgent
from standalone_user_proxy_agent import StandAloneUserProxyAgent

import autogen

config_list = autogen.config_list_from_json(
    "../../../../OAI_CONFIG_LIST", filter_dict={"model": ["gpt-3.5-turbo-1106"]}
)


def get_resource(description, save_path) -> str:
    print(f">>>> Calling get_resource({description}, {save_path})")
    f = open(save_path, "a")
    f.write(description)
    f.close()
    return os.path.abspath(save_path)


functions = [
    {
        "name": "get_resource",
        "description": "retrieves the requested resource like image, audio, video based on the description and save in save_path, return the abspath of file",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "a long fully description of resource to download, including type of resouce",
                },
                "save_path": {"type": "string", "description": "the path of file to save resource in"},
            },
        },
    }
]

function_map = {
    "get_resource": get_resource,
}

llm_config_local_llma = {"config_list": config_list, "temperature": 0.5, "max_retries": 20, "timeout": 300}

pm = StandAloneAssistantAgent(
    name="ProductManager",
    system_message="Creative in software product ideas.",
    sa_agent_config={
        "central_server": "http://localhost:8888",
        "port": 3333,
    },
    llm_config=llm_config_local_llma,
)

pm.serve()
pm.wait()
