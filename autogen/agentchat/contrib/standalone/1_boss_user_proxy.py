import os
import time

from standalone_assistant_agent import StandAloneAssistantAgent
from standalone_user_proxy_agent import StandAloneUserProxyAgent
from termcolor import colored

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
    human_input_mode="TERMINATE",
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

    user_proxy.initiate_chat(
        recipient="ChatManager",
        clear_history=True,
        message="""create a battleship classic game""",
    )
    user_proxy.wait()
except Exception as error:
    print(colored(error, "red"), flush=True)
    user_proxy.stop()
