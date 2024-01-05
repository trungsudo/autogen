import logging
import random
import sys
from typing import Dict, List, Optional, Tuple, Union

import requests
from central_server import AgentInfo, RunningState
from standalone_agent import StandAloneAgent
from termcolor import colored

from autogen import Agent, ConversableAgent, GroupChat, GroupChatManager

logger = logging.getLogger(__name__)


def get_agentinfo_from_name(central_server, agent_name: str) -> AgentInfo:
    params = {"name": agent_name}
    response = requests.get(central_server + "/agent", params=params)
    if response.status_code != 200:
        raise ValueError(response.content)
    info = AgentInfo(**response.json())
    print(
        colored(f"{agent_name} is {info.status}", "green" if info.status == RunningState.ONLINE else "red"), flush=True
    )
    if info.status != RunningState.ONLINE:
        raise ValueError(f"{agent_name} is {info.status}")
    return info


class AgentWrapper(Agent):
    def __init__(self, central_server: str, info: AgentInfo):
        self._central_server = central_server
        self._name = info.name
        self._info = info
        self._function_can_call = (
            None if info.functions_can_call is None else [func["name"] for func in info.functions_can_call]
        )
        self.function_map = info.functions_can_execute
        self.description = info.description

    def can_execute_function(self, name: str) -> bool:
        """Whether the agent can execute the function."""
        return name in self.function_map

    def generate_reply(self, sender: Agent):
        destination = self._info.endpoint + "/request_generate_reply"
        logger.debug(colored(f"{sender._name} is requesting {self._name} to reply, at address: {destination}", "blue"))
        response = requests.post(destination, json={"reply_to": sender._name})
        return response.json()

    @property
    def name(self):
        return self._name


class StandAloneGroupChat(GroupChat):
    def __init__(self, agents: Union[List[Agent], List[str]], sa_agent_config: Optional[Dict] = None, *args, **kwargs):
        if "central_server" not in sa_agent_config:
            raise ValueError("config is not contains central_server")
        self._unregistered_when_offline = True
        self._central_server = sa_agent_config["central_server"]

        _agents = [
            agent
            if isinstance(agent, Agent)
            else AgentWrapper(self._central_server, get_agentinfo_from_name(self._central_server, agent))
            for agent in agents
        ]
        super().__init__(agents=_agents, *args, **kwargs)


class StandAloneGroupChatManager(StandAloneAgent):
    """(In preview) A chat manager agent that can manage a group chat of multiple agents."""

    def __init__(
        self,
        groupchat: GroupChat,
        name: Optional[str] = "chat_manager",
        # unlimited consecutive auto reply by default
        max_consecutive_auto_reply: Optional[int] = sys.maxsize,
        human_input_mode: Optional[str] = "NEVER",
        system_message: Optional[Union[str, List]] = "Group chat manager.",
        **kwargs,
    ):
        if kwargs.get("llm_config") and (kwargs["llm_config"].get("functions") or kwargs["llm_config"].get("tools")):
            raise ValueError(
                "GroupChatManager is not allowed to make function/tool calls. Please remove the 'functions' or 'tools' config in 'llm_config' you passed in."
            )
        super().__init__(
            name=name,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            system_message=system_message,
            **kwargs,
        )
        # Order of register_reply is important.
        # Allow sync chat if initiated using initiate_chat
        self.register_reply(Agent, StandAloneGroupChatManager.run_chat, config=groupchat, reset_config=GroupChat.reset)
        # Allow async chat if initiated using a_initiate_chat
        self.register_reply(
            Agent, StandAloneGroupChatManager.a_run_chat, config=groupchat, reset_config=GroupChat.reset
        )

    def run_chat(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[GroupChat] = None,
    ) -> Union[str, Dict, None]:
        """Run a group chat."""
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        speaker = sender
        groupchat = config
        for i in range(groupchat.max_round):
            # set the name to speaker's name if the role is not function
            if message["role"] != "function":
                message["name"] = speaker.name

            groupchat.append(message)

            if self._is_termination_msg(message):
                # The conversation is over
                break
            # broadcast the message to all agents except the speaker
            for agent in groupchat.agents:
                if agent._name != speaker._name:
                    self.send(message, agent, request_reply=False, silent=True)
            if i == groupchat.max_round - 1:
                # the last round
                break
            try:
                # select the next speaker
                speaker = groupchat.select_speaker(speaker, self)
                logger.debug(colored(f">>>> NEXT SPEAkER {speaker._name}", "light_cyan"))
                # let the speaker speak
                reply = speaker.generate_reply(sender=self)
            except KeyboardInterrupt:
                # let the admin agent speak if interrupted
                if groupchat.admin_name in groupchat.agent_names:
                    # admin agent is one of the participants
                    speaker = groupchat.agent_by_name(groupchat.admin_name)
                    reply = speaker.generate_reply(sender=self)
                else:
                    # admin agent is not found in the participants
                    raise
            if reply is None:
                break
            # The speaker sends the message without requesting a reply
            # speaker.send(reply, self, request_reply=False)
            self._process_received_message(str(reply), speaker, True)
            message = self.last_message(speaker)
        return True, None

    async def a_run_chat(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[GroupChat] = None,
    ):
        """Run a group chat asynchronously."""
        self.run_chat(messages=messages, sender=sender, config=config)
