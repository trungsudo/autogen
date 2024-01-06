import os

from termcolor import colored

import autogen
from autogen.autogen.agentchat.contrib.independent_agent.i_groupchat import (
    IndependentGroupChat,
    IndependentGroupChatManager,
)

config_list = autogen.config_list_from_json("../../../../OAI_CONFIG_LIST", filter_dict={"model": ["llama-v2"]})
# config_list = autogen.config_list_from_json("autogen/OAI_CONFIG_LIST", filter_dict={"model": ["gpt-3.5-turbo-1106"]})
llm_config = {"config_list": config_list, "temperature": 0.5, "max_retries": 20, "timeout": 300}
sa_agent_config = {
    "central_server": "http://localhost:8888",
    "port": 4444,
}
groupchat = IndependentGroupChat(
    agents=["Boss", "Coder", "ProductManager"],
    allow_repeat_speaker=False,
    sa_agent_config=sa_agent_config,
    messages=[],
    max_round=12,
)

chat_manager = IndependentGroupChatManager(
    name="ChatManager",
    groupchat=groupchat,
    llm_config=llm_config,
    sa_agent_config=sa_agent_config,
)

try:
    chat_manager.serve()
    chat_manager.main_loop()
except Exception as error:
    print(colored(error, "red"), flush=True)
    chat_manager.stop()
