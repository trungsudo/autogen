import os

from standalone_assistant_agent import StandAloneAssistantAgent
from standalone_user_proxy_agent import StandAloneUserProxyAgent

import autogen
from autogen import ConversableAgent


def get_resource(description, save_path) -> str:
    print(f">>>> Calling get_resource({description}, {save_path})")
    f = open(save_path, "a")
    f.write(description)
    f.close()
    return os.path.abspath(save_path)


function_map = {"get_resource": get_resource}

# create a UserProxyAgent instance named "user_proxy"
user_proxy = StandAloneUserProxyAgent(
    name="Boss",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=12,
    # is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },
    function_map=function_map,
    agent_config={
        "central_server": "http://localhost:8888",
        "host": "localhost",
        "port": 5432,
    },
)

user_proxy.register_function(function_map=function_map)

user_proxy.serve()

# the assistant receives a message from the user_proxy, which contains the task description
user_proxy.initiate_chat(
    ConversableAgent(name="Developer", llm_config=False),
    clear_history=True,
    message="""What is our client name, again?""",
)

user_proxy.wait()
