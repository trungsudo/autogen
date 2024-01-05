import os

from standalone_groupchat import StandAloneGroupChat, StandAloneGroupChatManager

import autogen

config_list = autogen.config_list_from_json(
    "../../../../OAI_CONFIG_LIST", filter_dict={"model": ["gpt-3.5-turbo-1106"]}
)
# config_list = autogen.config_list_from_json("autogen/OAI_CONFIG_LIST", filter_dict={"model": ["gpt-3.5-turbo-1106"]})
llm_config = {"config_list": config_list, "temperature": 0.5, "max_retries": 20, "timeout": 300}
sa_agent_config = {
    "central_server": "http://localhost:8888",
    "port": 4444,
}
groupchat = StandAloneGroupChat(
    agents=["Boss", "Developer", "ProductManager"], sa_agent_config=sa_agent_config, messages=[], max_round=12
)

chat_manager = StandAloneGroupChatManager(
    name="ChatManager",
    groupchat=groupchat,
    llm_config=llm_config,
    sa_agent_config=sa_agent_config,
)

try:
    chat_manager.serve()
    chat_manager.wait()
except:
    chat_manager.stop()
