import os
import time

from termcolor import colored

import autogen
from autogen import ConversableAgent
from autogen.autogen.agentchat.contrib.independent_agent.i_assistant_agent import (
    IndependentAssistantAgent,
)
from autogen.autogen.agentchat.contrib.independent_agent.i_user_proxy_agent import (
    IndependentUserProxyAgent,
)


def get_resource(description, save_path) -> str:
    print(f">>>> Calling get_resource({description}, {save_path})")
    f = open(save_path, "a")
    f.write(description)
    f.close()
    return os.path.abspath(save_path)


function_map = {"get_resource": get_resource}

user_proxy = IndependentUserProxyAgent(
    name="Boss",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=12,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },
    function_map=function_map,
    sa_agent_config={
        "central_server": "http://localhost:8888",
        "host": "localhost",
        "port": 1111,
    },
)

user_proxy.register_function(function_map=function_map)
try:
    user_proxy.serve()
    # TODO: Get notification from server
    user_proxy.wait_for_agent("ChatManager")
    print("Send request to ChatManager")
    user_proxy.initiate_chat(
        recipient="ChatManager",
        clear_history=True,
        message="""create a GUI classic game. python code is provided in a python code block: 
        ```python
        ...
        ```
        """,
    )
    user_proxy.main_loop()
except Exception as error:
    print(colored(error, "red"), flush=True)
    user_proxy.stop()
