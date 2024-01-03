import argparse
import asyncio
import copy
import functools
import json
import logging
import signal
import socket
import sys
import threading
import time
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import requests
from central_server import AgentInfo, RunningState
from flask import Flask, request
from pydantic import BaseModel
from termcolor import colored
from werkzeug.serving import make_server

from autogen import Agent, ConversableAgent, config_list_from_json
from autogen.code_utils import UNKNOWN, extract_code
from autogen.oai import OpenAIWrapper

logger = logging.getLogger(__name__)


class Message(BaseModel):
    sender_name: str
    receiver_name: str
    message: Union[Dict, str]
    request_reply: bool = False
    silent: bool = False


class StandAloneAgent(ConversableAgent):
    def __init__(self, agent_config: Optional[Dict] = None, *args, **kwargs):
        if "port" not in agent_config or "central_server" not in agent_config:
            raise Exception("config is not contains port/central_server")
        self._port = int(agent_config.get("port", 1234))
        self.central_server = agent_config["central_server"]
        self._host = agent_config["host"] if "host" in agent_config else socket.gethostbyname(socket.gethostname())
        agent_config.pop("port")
        agent_config.pop("central_server")
        if "host" in agent_config:
            agent_config.pop("host")

        super().__init__(*args, **agent_config, **kwargs)

        self._register_agent()
        self.register_reply([Agent, None], StandAloneAgent.generate_oai_reply)
        self.register_reply([Agent, None], StandAloneAgent.a_generate_oai_reply)
        self.register_reply([Agent, None], StandAloneAgent.generate_code_execution_reply)
        self.register_reply([Agent, None], StandAloneAgent.generate_function_call_reply)
        self.register_reply([Agent, None], StandAloneAgent.a_generate_function_call_reply)
        self.register_reply([Agent, None], StandAloneAgent.check_termination_and_human_reply)
        self.register_reply([Agent, None], StandAloneAgent.a_check_termination_and_human_reply)

        self.app = Flask(__name__)
        self.server = make_server(self._host, self._port, self.app)

        @self.app.route("/message", methods=["POST"])
        def receive_http_message():
            message = Message(**request.get_json())
            print(colored(f"{message.sender_name} -> {self.name}: {message.message}", "green"), flush=True)
            thread = threading.Thread(target=self.handler, args=(message,))
            thread.start()
            return "Message received", 200

        @self.app.route("/request_reset_consecutive_auto_reply_counter", methods=["POST"])
        def handle_request_reset_consecutive_auto_reply_counter():
            """Reset the consecutive_auto_reply_counter of the sender HTTP handler."""
            data = request.get_json(force=True)
            self.reset_consecutive_auto_reply_counter(data.get("name"))

        @self.app.route("/request_reply_at_receive", methods=["POST"])
        def handle_request_reply_at_receive():
            data = request.get_json(force=True)
            if data is not None and data["name"] is not None and data["request_reply"] is not None:
                self.reply_at_receive[data["name"]] = data["request_reply"]
                print(f"reply_at_receive[{data['name']}]: {self.reply_at_receive[data['name']]}")

    def request_reply_at_receive(self, recipient: Agent, request_reply: bool):
        destination = self.get_agentinfo_from_name(recipient.name).endpoint + "/request_reply_at_receive"
        headers = {"Content-Type": "application/json"}
        requests.post(destination, headers=headers, json={"name": {self.name}, "request_reply": request_reply})

    def request_reset_consecutive_auto_reply_counter(self, recipient: Agent):
        """Send request to reset the consecutive_auto_reply_counter of the receiver with sender"""
        destination = (
            self.get_agentinfo_from_name(recipient.name).endpoint + "/request_reset_consecutive_auto_reply_counter"
        )
        headers = {"Content-Type": "application/json"}
        requests.post(destination, headers=headers, json=f'{"name": {self.name}}')

    def handler(self, message: Message):
        self.receive(
            message=message.message,
            sender=Agent(message.sender_name),
            request_reply=message.request_reply,
            silent=message.silent,
        )

    def get_agentinfo_from_name(self, agent_name: str) -> AgentInfo:
        params = {"name": agent_name}
        response = requests.get(self.central_server + "/agent", params=params)

        if response.status_code != 200:
            raise Exception(response.json()["error"])
        status = AgentInfo(**response.json())
        return status

    def _register_agent(self):
        message = AgentInfo(
            name=self._name, description=self.system_message, endpoint=f"http://{self._host}:{self._port}"
        )
        response = requests.post(self.central_server + "/register", json=message.model_dump())
        if response.status_code != 200:
            raise Exception(response.content)

    def _self_unregister(self):
        requests.post(self.central_server + "/self_unregister", json={"name": self._name})

    def serve(self):
        print(colored(f"{self._name} is ONLINE at {self._host}:{self._port}", "green"))
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()

    def stop(self):
        try:
            response = self._self_unregister()
        except Exception as error:
            print(error)
        print(colored(f"{self._name} is OFFLINE", "red"))
        self.server.shutdown()
        self.thread.join()

    def send_http_message(self, message: Message, destination: str):
        headers = {"Content-Type": "application/json"}
        print(f"Send to {destination}: {message.model_dump_json()}")
        response = requests.post(destination, headers=headers, json=message.model_dump())
        print(f"Got response {response.content}")
        return response.status_code

    ####################################################################

    # def register_reply(
    #     self,
    #     trigger: Union[Type[Agent], str, Agent, Callable[[Agent], bool], List],
    #     reply_func: Callable,
    #     position: int = 0,
    #     config: Optional[Any] = None,
    #     reset_config: Optional[Callable] = None,
    # )

    # def system_message(self) -> Union[str, List]

    # def update_system_message(self, system_message: Union[str, List])

    def update_max_consecutive_auto_reply(self, value: int, sender: Optional[Agent] = None):
        """Update the maximum number of consecutive auto replies.

        Args:
            value (int): the maximum number of consecutive auto replies.
            sender (Agent): when the sender is provided, only update the max_consecutive_auto_reply for that sender.
        """
        if sender is None:
            self._max_consecutive_auto_reply = value
            for k in self._max_consecutive_auto_reply_dict:
                self._max_consecutive_auto_reply_dict[k] = value
        else:
            self._max_consecutive_auto_reply_dict[sender.name] = value

    def max_consecutive_auto_reply(self, sender: Optional[Agent] = None) -> int:
        """The maximum number of consecutive auto replies."""
        return (
            self._max_consecutive_auto_reply if sender is None else self._max_consecutive_auto_reply_dict[sender.name]
        )

    # Broken: def chat_messages(self) -> Dict[Agent, List[Dict]]
    @property
    def chat_messages(self) -> Dict[str, List[Dict]]:
        """A dictionary of conversations from agent to list of messages."""
        return self._oai_messages

    def last_message(self, agent: Optional[Agent] = None) -> Optional[Dict]:
        """The last message exchanged with the agent.

        Args:
            agent (Agent): The agent in the conversation.
                If None and more than one agent's conversations are found, an error will be raised.
                If None and only one conversation is found, the last message of the only conversation will be returned.

        Returns:
            The last message exchanged with the agent.
        """
        if agent is None:
            n_conversations = len(self._oai_messages)
            if n_conversations == 0:
                return None
            if n_conversations == 1:
                for conversation in self._oai_messages.values():
                    return conversation[-1]
            raise ValueError("More than one conversation is found. Please specify the sender to get the last message.")
        if agent.name not in self._oai_messages.keys():
            raise KeyError(
                f"The agent '{agent.name}' is not present in any conversation. No history available for this agent."
            )
        return self._oai_messages[agent.name][-1]

    # def use_docker(self) -> Union[bool, str, None]

    # def _message_to_dict(message: Union[Dict, str]) -> Dict

    def _append_oai_message(self, message: Union[Dict, str], role, conversation_id: Agent) -> bool:
        """Append a message to the ChatCompletion conversation.

        If the message received is a string, it will be put in the "content" field of the new dictionary.
        If the message received is a dictionary but does not have any of the two fields "content" or "function_call",
            this message is not a valid ChatCompletion message.
        If only "function_call" is provided, "content" will be set to None if not provided, and the role of the message will be forced "assistant".

        Args:
            message (dict or str): message to be appended to the ChatCompletion conversation.
            role (str): role of the message, can be "assistant" or "function".
            conversation_id (Agent): id of the conversation, should be the recipient or sender.

        Returns:
            bool: whether the message is appended to the ChatCompletion conversation.
        """
        message = self._message_to_dict(message)
        # create oai message to be appended to the oai conversation that can be passed to oai directly.
        oai_message = {k: message[k] for k in ("content", "function_call", "name", "context") if k in message}
        if "content" not in oai_message:
            if "function_call" in oai_message:
                oai_message["content"] = None  # if only function_call is provided, content will be set to None.
            else:
                return False

        oai_message["role"] = "function" if message.get("role") == "function" else role
        if "function_call" in oai_message:
            oai_message["role"] = "assistant"  # only messages with role 'assistant' can have a function call.
            oai_message["function_call"] = dict(oai_message["function_call"])
        self._oai_messages[conversation_id.name].append(oai_message)
        return True

    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        """Send a message to another agent.

        Args:
            message (dict or str): message to be sent.
                The message could contain the following fields:
                - content (str or List): Required, the content of the message. (Can be None)
                - function_call (str): the name of the function to be called.
                - name (str): the name of the function to be called.
                - role (str): the role of the message, any role that is not "function"
                    will be modified to "assistant".
                - context (dict): the context of the message, which will be passed to
                    [OpenAIWrapper.create](../oai/client#create).
                    For example, one agent can send a message A as:
        ```python
        {
            "content": lambda context: context["use_tool_msg"],
            "context": {
                "use_tool_msg": "Use tool X if they are relevant."
            }
        }
        ```
                    Next time, one agent can send a message B with a different "use_tool_msg".
                    Then the content of message A will be refreshed to the new "use_tool_msg".
                    So effectively, this provides a way for an agent to send a "link" and modify
                    the content of the "link" later.
            recipient (Agent): the recipient of the message.
            request_reply (bool or None): whether to request a reply from the recipient.
            silent (bool or None): (Experimental) whether to print the message sent.

        Raises:
            ValueError: if the message can't be converted into a valid ChatCompletion message.
            Exception: can not get agentinfo from its name
        """
        valid = self._append_oai_message(message, "assistant", recipient)

        if valid:
            # recipient.receive(message, self, request_reply, silent)
            try:
                target_url = self.get_agentinfo_from_name(recipient.name).endpoint + "/message"
            except Exception as e:
                raise e
            http_message = Message(
                sender_name=self._name,
                receiver_name=recipient.name,
                message=message,
                request_reply=request_reply,
                silent=silent,
            )
            print(colored(f"{self.name} -> {recipient.name}: {message}", "light_red"), flush=True)
            self.send_http_message(http_message, target_url)
        else:
            raise ValueError(
                "Message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            )

    async def a_send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        """(async) Send a message to another agent.

        Args:
            message (dict or str): message to be sent.
                The message could contain the following fields:
                - content (str or List): Required, the content of the message. (Can be None)
                - function_call (str): the name of the function to be called.
                - name (str): the name of the function to be called.
                - role (str): the role of the message, any role that is not "function"
                    will be modified to "assistant".
                - context (dict): the context of the message, which will be passed to
                    [OpenAIWrapper.create](../oai/client#create).
                    For example, one agent can send a message A as:
        ```python
        {
            "content": lambda context: context["use_tool_msg"],
            "context": {
                "use_tool_msg": "Use tool X if they are relevant."
            }
        }
        ```
                    Next time, one agent can send a message B with a different "use_tool_msg".
                    Then the content of message A will be refreshed to the new "use_tool_msg".
                    So effectively, this provides a way for an agent to send a "link" and modify
                    the content of the "link" later.
            recipient (Agent): the recipient of the message.
            request_reply (bool or None): whether to request a reply from the recipient.
            silent (bool or None): (Experimental) whether to print the message sent.

        Raises:
            ValueError: if the message can't be converted into a valid ChatCompletion message.
        """
        # return await asyncio.get_event_loop().run_in_executor(
        #     None, functools.partial(self.generate_oai_reply, messages=messages, sender=sender, config=config)
        # )
        self.send(message, recipient, request_reply, silent)

    # def _print_received_message(self, message: Union[Dict, str], sender: Agent)

    # def _process_received_message(self, message: Union[Dict, str], sender: Agent, silent: bool)

    def receive(
        self,
        message: Union[Dict, str],
        sender: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        """Receive a message from another agent.

        Once a message is received, this function sends a reply to the sender or stop.
        The reply can be generated automatically or entered manually by a human.

        Args:
            message (dict or str): message from the sender. If the type is dict, it may contain the following reserved fields (either content or function_call need to be provided).
                1. "content": content of the message, can be None.
                2. "function_call": a dictionary containing the function name and arguments.
                3. "role": role of the message, can be "assistant", "user", "function".
                    This field is only needed to distinguish between "function" or "assistant"/"user".
                4. "name": In most cases, this field is not needed. When the role is "function", this field is needed to indicate the function name.
                5. "context" (dict): the context of the message, which will be passed to
                    [OpenAIWrapper.create](../oai/client#create).
            sender: sender of an Agent instance.
            request_reply (bool or None): whether a reply is requested from the sender.
                If None, the value is determined by `self.reply_at_receive[sender]`.
            silent (bool or None): (Experimental) whether to print the message received.

        Raises:
            ValueError: if the message can't be converted into a valid ChatCompletion message.
        """
        self._process_received_message(message, sender, silent)
        if request_reply is False or request_reply is None and self.reply_at_receive[sender.name] is False:
            print(colored(f"{self.name} is not gonna reply to {sender.name}", "red"), flush=True)
            return

        reply = self.generate_reply(messages=self.chat_messages[sender], sender=sender)
        if reply is not None:
            # self.send(reply, sender, silent=silent)
            try:
                target_url = self.get_agentinfo_from_name(sender.name).endpoint
            except Exception as e:
                print(e)
                return
            self.send(message=reply, recipient=sender, request_reply=True, silent=silent)
        else:
            print(colored(f"{self.name} has nothing to tell with {sender.name}", "red"), flush=True)

    async def a_receive(
        self,
        message: Union[Dict, str],
        sender: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        # return await asyncio.get_event_loop().run_in_executor(
        #     None, functools.partial(self.generate_oai_reply, messages=messages, sender=sender, config=config)
        # )
        self.receive(message, sender, request_reply, silent)

    def _prepare_chat(self, recipient, clear_history):
        self.reset_consecutive_auto_reply_counter(recipient.name)
        self.request_reset_consecutive_auto_reply_counter(recipient)

        self.reply_at_receive[recipient.name] = True
        self.request_reply_at_receive(recipient, True)

        if clear_history:
            self.clear_history(recipient)
            recipient.clear_history(self)

    def initiate_chat(
        self,
        recipient: "ConversableAgent",
        clear_history: Optional[bool] = True,
        silent: Optional[bool] = False,
        **context,
    ):
        """Initiate a chat with the recipient agent.

        Reset the consecutive auto reply counter.
        If `clear_history` is True, the chat history with the recipient agent will be cleared.
        `generate_init_message` is called to generate the initial message for the agent.

        Args:
            recipient: the recipient agent.
            clear_history (bool): whether to clear the chat history with the agent.
            silent (bool or None): (Experimental) whether to print the messages for this conversation.
            **context: any context information.
                "message" needs to be provided if the `generate_init_message` method is not overridden.
        """
        self._prepare_chat(recipient, clear_history)
        self.send(self.generate_init_message(**context), recipient, silent=silent)

    async def a_initiate_chat(
        self,
        recipient: "ConversableAgent",
        clear_history: Optional[bool] = True,
        silent: Optional[bool] = False,
        **context,
    ):
        """(async) Initiate a chat with the recipient agent.

        Reset the consecutive auto reply counter.
        If `clear_history` is True, the chat history with the recipient agent will be cleared.
        `generate_init_message` is called to generate the initial message for the agent.

        Args:
            recipient: the recipient agent.
            clear_history (bool): whether to clear the chat history with the agent.
            silent (bool or None): (Experimental) whether to print the messages for this conversation.
            **context: any context information.
                "message" needs to be provided if the `generate_init_message` method is not overridden.
        """
        self._prepare_chat(recipient, clear_history)
        await self.a_send(self.generate_init_message(**context), recipient, silent=silent)

    def reset(self):
        """Reset the agent."""
        self.clear_history()
        self.reset_consecutive_auto_reply_counter()
        self.stop_reply_at_receive()
        for reply_func_tuple in self._reply_func_list:
            if reply_func_tuple["reset_config"] is not None:
                reply_func_tuple["reset_config"](reply_func_tuple["config"])
            else:
                reply_func_tuple["config"] = copy.copy(reply_func_tuple["init_config"])

    def stop_reply_at_receive(self, sender: Optional[Agent] = None):
        """Reset the reply_at_receive of the sender."""
        if sender is None:
            self.reply_at_receive.clear()
        else:
            self.reply_at_receive[sender.name] = False

    def reset_consecutive_auto_reply_counter(self, name: str):
        """Reset the consecutive_auto_reply_counter of the sender."""
        if name is not None:
            self._consecutive_auto_reply_counter[name] = 0
        else:
            self._consecutive_auto_reply_counter.clear()

    def clear_history(self, agent: Optional[Agent] = None):
        """Clear the chat history of the agent.

        Args:
            agent: the agent with whom the chat history to clear. If None, clear the chat history with all agents.
        """
        if agent is None:
            self._oai_messages.clear()
        else:
            self._oai_messages[agent.name].clear()

    def generate_oai_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[OpenAIWrapper] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        """Generate a reply using autogen.oai."""
        client = self.client if config is None else config
        if client is None:
            return False, None
        if messages is None or len(messages) == 0:
            messages = self._oai_messages[sender.name]

        # TODO: #1143 handle token limit exceeded error
        ctx = messages[-1].pop("context", None)
        msg = self._oai_system_message + messages
        print(
            colored(
                f"{self.name} is using OAI, based on:\n-Context: {ctx}\n-Messages: {json.dumps(msg, indent=4)}",
                "yellow",
            ),
            flush=True,
        )
        response = client.create(context=ctx, messages=msg)

        # TODO: line 301, line 271 is converting messages to dict. Can be removed after ChatCompletionMessage_to_dict is merged.
        extracted_response = client.extract_text_or_completion_object(response)[0]
        if not isinstance(extracted_response, str):
            extracted_response = extracted_response.model_dump(mode="dict")
        return True, extracted_response

    async def a_generate_oai_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        """Generate a reply using autogen.oai asynchronously."""
        return await asyncio.get_event_loop().run_in_executor(
            None, functools.partial(self.generate_oai_reply, messages=messages, sender=sender, config=config)
        )

    def generate_code_execution_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Union[Dict, Literal[False]]] = None,
    ):
        """Generate a reply using code execution."""
        print(colored(f"generate_code_execution_reply", "blue"), flush=True)
        code_execution_config = config if config is not None else self._code_execution_config
        if code_execution_config is False:
            return False, None
        if messages is None:
            messages = self._oai_messages[sender.name]
        last_n_messages = code_execution_config.pop("last_n_messages", 1)

        messages_to_scan = last_n_messages
        if last_n_messages == "auto":
            # Find when the agent last spoke
            messages_to_scan = 0
            for i in range(len(messages)):
                message = messages[-(i + 1)]
                if "role" not in message:
                    break
                elif message["role"] != "user":
                    break
                else:
                    messages_to_scan += 1

        # iterate through the last n messages reversely
        # if code blocks are found, execute the code blocks and return the output
        # if no code blocks are found, continue
        for i in range(min(len(messages), messages_to_scan)):
            message = messages[-(i + 1)]
            if not message["content"]:
                continue
            code_blocks = extract_code(message["content"])
            if len(code_blocks) == 1 and code_blocks[0][0] == UNKNOWN:
                continue

            # found code blocks, execute code and push "last_n_messages" back
            exitcode, logs = self.execute_code_blocks(code_blocks)
            code_execution_config["last_n_messages"] = last_n_messages
            exitcode2str = "execution succeeded" if exitcode == 0 else "execution failed"
            return True, f"exitcode: {exitcode} ({exitcode2str})\nCode output: {logs}"

        # no code blocks are found, push last_n_messages back and return.
        code_execution_config["last_n_messages"] = last_n_messages

        return False, None

    def generate_function_call_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[Dict, None]]:
        """Generate a reply using function call."""
        print(colored(f"generate_function_call_reply", "blue"), flush=True)
        if config is None:
            config = self
        if messages is None or len(messages) == 0:
            messages = self._oai_messages[sender.name]
        message = messages[-1]
        if "function_call" in message:
            _, func_return = self.execute_function(message["function_call"])
            return True, func_return
        return False, None

    async def a_generate_function_call_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[Dict, None]]:
        """Generate a reply using async function call."""
        if config is None:
            config = self
        if messages is None or len(messages) == 0:
            messages = self._oai_messages[sender.name]
        message = messages[-1]
        if "function_call" in message:
            func_call = message["function_call"]
            func_name = func_call.get("name", "")
            func = self._function_map.get(func_name, None)
            if func and asyncio.coroutines.iscoroutinefunction(func):
                _, func_return = await self.a_execute_function(func_call)
                return True, func_return

        return False, None

    def check_termination_and_human_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, None]]:
        """Check if the conversation should be terminated, and if human reply is provided.

        This method checks for conditions that require the conversation to be terminated, such as reaching
        a maximum number of consecutive auto-replies or encountering a termination message. Additionally,
        it prompts for and processes human input based on the configured human input mode, which can be
        'ALWAYS', 'NEVER', or 'TERMINATE'. The method also manages the consecutive auto-reply counter
        for the conversation and prints relevant messages based on the human input received.

        Args:
            - messages (Optional[List[Dict]]): A list of message dictionaries, representing the conversation history.
            - sender (Optional[Agent]): The agent object representing the sender of the message.
            - config (Optional[Any]): Configuration object, defaults to the current instance if not provided.

        Returns:
            - Tuple[bool, Union[str, Dict, None]]: A tuple containing a boolean indicating if the conversation
            should be terminated, and a human reply which can be a string, a dictionary, or None.
        """
        # Function implementation...
        print(colored(f"check_termination_and_human_reply", "blue"), flush=True)
        if config is None:
            config = self
        if messages is None or len(messages) == 0:
            messages = self._oai_messages[sender.name]
        message = messages[-1]
        reply = ""
        no_human_input_msg = ""
        if self.human_input_mode == "ALWAYS":
            reply = self.get_human_input(
                f"Provide feedback to {sender.name}. Press enter to skip and use auto-reply, or type 'exit' to end the conversation: "
            )
            no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
            # if the human input is empty, and the message is a termination message, then we will terminate the conversation
            reply = reply if reply or not self._is_termination_msg(message) else "exit"
        else:
            if self._consecutive_auto_reply_counter[sender.name] >= self._max_consecutive_auto_reply_dict[sender.name]:
                if self.human_input_mode == "NEVER":
                    reply = "exit"
                else:
                    # self.human_input_mode == "TERMINATE":
                    terminate = self._is_termination_msg(message)
                    reply = self.get_human_input(
                        f"Please give feedback to {sender.name}. Press enter or type 'exit' to stop the conversation: "
                        if terminate
                        else f"Please give feedback to {sender.name}. Press enter to skip and use auto-reply, or type 'exit' to stop the conversation: "
                    )
                    no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
                    # if the human input is empty, and the message is a termination message, then we will terminate the conversation
                    reply = reply if reply or not terminate else "exit"
            elif self._is_termination_msg(message):
                if self.human_input_mode == "NEVER":
                    reply = "exit"
                else:
                    # self.human_input_mode == "TERMINATE":
                    reply = self.get_human_input(
                        f"Please give feedback to {sender.name}. Press enter or type 'exit' to stop the conversation: "
                    )
                    no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
                    # if the human input is empty, and the message is a termination message, then we will terminate the conversation
                    reply = reply or "exit"

        # print the no_human_input_msg
        if no_human_input_msg:
            print(colored(f"\n>>>>>>>> {no_human_input_msg}", "red"), flush=True)

        # stop the conversation
        if reply == "exit":
            # reset the consecutive_auto_reply_counter
            self._consecutive_auto_reply_counter[sender.name] = 0
            return True, None

        # send the human reply
        if reply or self._max_consecutive_auto_reply_dict[sender.name] == 0:
            # reset the consecutive_auto_reply_counter
            self._consecutive_auto_reply_counter[sender.name] = 0
            return True, reply

        # increment the consecutive_auto_reply_counter
        self._consecutive_auto_reply_counter[sender.name] += 1
        if self.human_input_mode != "NEVER":
            print(colored("\n>>>>>>>> USING AUTO REPLY...", "red"), flush=True)

        return False, None

    async def a_check_termination_and_human_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, None]]:
        """(async) Check if the conversation should be terminated, and if human reply is provided.

        This method checks for conditions that require the conversation to be terminated, such as reaching
        a maximum number of consecutive auto-replies or encountering a termination message. Additionally,
        it prompts for and processes human input based on the configured human input mode, which can be
        'ALWAYS', 'NEVER', or 'TERMINATE'. The method also manages the consecutive auto-reply counter
        for the conversation and prints relevant messages based on the human input received.

        Args:
            - messages (Optional[List[Dict]]): A list of message dictionaries, representing the conversation history.
            - sender (Optional[Agent]): The agent object representing the sender of the message.
            - config (Optional[Any]): Configuration object, defaults to the current instance if not provided.

        Returns:
            - Tuple[bool, Union[str, Dict, None]]: A tuple containing a boolean indicating if the conversation
            should be terminated, and a human reply which can be a string, a dictionary, or None.
        """
        if config is None:
            config = self
        if messages is None or len(messages) == 0:
            messages = self._oai_messages[sender.name]
        message = messages[-1]
        reply = ""
        no_human_input_msg = ""
        if self.human_input_mode == "ALWAYS":
            reply = await self.a_get_human_input(
                f"Provide feedback to {sender.name}. Press enter to skip and use auto-reply, or type 'exit' to end the conversation: "
            )
            no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
            # if the human input is empty, and the message is a termination message, then we will terminate the conversation
            reply = reply if reply or not self._is_termination_msg(message) else "exit"
        else:
            if self._consecutive_auto_reply_counter[sender.name] >= self._max_consecutive_auto_reply_dict[sender.name]:
                if self.human_input_mode == "NEVER":
                    reply = "exit"
                else:
                    # self.human_input_mode == "TERMINATE":
                    terminate = self._is_termination_msg(message)
                    reply = await self.a_get_human_input(
                        f"Please give feedback to {sender.name}. Press enter or type 'exit' to stop the conversation: "
                        if terminate
                        else f"Please give feedback to {sender.name}. Press enter to skip and use auto-reply, or type 'exit' to stop the conversation: "
                    )
                    no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
                    # if the human input is empty, and the message is a termination message, then we will terminate the conversation
                    reply = reply if reply or not terminate else "exit"
            elif self._is_termination_msg(message):
                if self.human_input_mode == "NEVER":
                    reply = "exit"
                else:
                    # self.human_input_mode == "TERMINATE":
                    reply = await self.a_get_human_input(
                        f"Please give feedback to {sender.name}. Press enter or type 'exit' to stop the conversation: "
                    )
                    no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
                    # if the human input is empty, and the message is a termination message, then we will terminate the conversation
                    reply = reply or "exit"

        # print the no_human_input_msg
        if no_human_input_msg:
            print(colored(f"\n>>>>>>>> {no_human_input_msg}", "red"), flush=True)

        # stop the conversation
        if reply == "exit":
            # reset the consecutive_auto_reply_counter
            self._consecutive_auto_reply_counter[sender.name] = 0
            return True, None

        # send the human reply
        if reply or self._max_consecutive_auto_reply_dict[sender.name] == 0:
            # reset the consecutive_auto_reply_counter
            self._consecutive_auto_reply_counter[sender.name] = 0
            return True, reply

        # increment the consecutive_auto_reply_counter
        self._consecutive_auto_reply_counter[sender.name] += 1
        if self.human_input_mode != "NEVER":
            print(colored("\n>>>>>>>> USING AUTO REPLY...", "red"), flush=True)

        return False, None

    def generate_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        exclude: Optional[List[Callable]] = None,
    ) -> Union[str, Dict, None]:
        """Reply based on the conversation history and the sender.

        Either messages or sender must be provided.
        Register a reply_func with `None` as one trigger for it to be activated when `messages` is non-empty and `sender` is `None`.
        Use registered auto reply functions to generate replies.
        By default, the following functions are checked in order:
        1. check_termination_and_human_reply
        2. generate_function_call_reply
        3. generate_code_execution_reply
        4. generate_oai_reply
        Every function returns a tuple (final, reply).
        When a function returns final=False, the next function will be checked.
        So by default, termination and human reply will be checked first.
        If not terminating and human reply is skipped, execute function or code and return the result.
        AI replies are generated only when no code execution is performed.

        Args:
            messages: a list of messages in the conversation history.
            default_reply (str or dict): default reply.
            sender: sender of an Agent instance.
            exclude: a list of functions to exclude.

        Returns:
            str or dict or None: reply. None if no reply is generated.
        """
        print(colored(f"{self.name} generate_reply", "blue"), flush=True)
        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            logger.error(error_msg)
            raise AssertionError(error_msg)

        if messages is None:
            messages = self._oai_messages[sender.name]

        for reply_func_tuple in self._reply_func_list:
            reply_func = reply_func_tuple["reply_func"]
            if exclude and reply_func in exclude:
                continue
            if asyncio.coroutines.iscoroutinefunction(reply_func):
                continue
            if self._match_trigger(reply_func_tuple["trigger"], sender):
                final, reply = reply_func(self, messages=messages, sender=sender, config=reply_func_tuple["config"])
                if final:
                    return reply
        return self._default_auto_reply

    async def a_generate_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        exclude: Optional[List[Callable]] = None,
    ) -> Union[str, Dict, None]:
        """(async) Reply based on the conversation history and the sender.

        Either messages or sender must be provided.
        Register a reply_func with `None` as one trigger for it to be activated when `messages` is non-empty and `sender` is `None`.
        Use registered auto reply functions to generate replies.
        By default, the following functions are checked in order:
        1. check_termination_and_human_reply
        2. generate_function_call_reply
        3. generate_code_execution_reply
        4. generate_oai_reply
        Every function returns a tuple (final, reply).
        When a function returns final=False, the next function will be checked.
        So by default, termination and human reply will be checked first.
        If not terminating and human reply is skipped, execute function or code and return the result.
        AI replies are generated only when no code execution is performed.

        Args:
            messages: a list of messages in the conversation history.
            default_reply (str or dict): default reply.
            sender: sender of an Agent instance.
            exclude: a list of functions to exclude.

        Returns:
            str or dict or None: reply. None if no reply is generated.
        """
        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            logger.error(error_msg)
            raise AssertionError(error_msg)

        if messages is None:
            messages = self._oai_messages[sender.name]

        for reply_func_tuple in self._reply_func_list:
            reply_func = reply_func_tuple["reply_func"]
            if exclude and reply_func in exclude:
                continue
            if self._match_trigger(reply_func_tuple["trigger"], sender):
                if asyncio.coroutines.iscoroutinefunction(reply_func):
                    final, reply = await reply_func(
                        self, messages=messages, sender=sender, config=reply_func_tuple["config"]
                    )
                else:
                    final, reply = reply_func(self, messages=messages, sender=sender, config=reply_func_tuple["config"])
                if final:
                    return reply
        return self._default_auto_reply

    # TODO: Review
    def _match_trigger(self, trigger: Union[None, str, type, Agent, Callable, List], sender: Agent) -> bool:
        """Check if the sender matches the trigger.

        Args:
            - trigger (Union[None, str, type, Agent, Callable, List]): The condition to match against the sender.
            Can be `None`, string, type, `Agent` instance, callable, or a list of these.
            - sender (Agent): The sender object or type to be matched against the trigger.

        Returns:
            - bool: Returns `True` if the sender matches the trigger, otherwise `False`.

        Raises:
            - ValueError: If the trigger type is unsupported.
        """
        if trigger is None:
            return sender is None
        elif isinstance(trigger, str):
            return trigger == sender.name
        elif isinstance(trigger, type):
            return isinstance(sender, trigger)
        elif isinstance(trigger, Agent):
            # return True if the sender is the same type (class) as the trigger
            return trigger == sender
        elif isinstance(trigger, Callable):
            rst = trigger(sender)
            assert rst in [True, False], f"trigger {trigger} must return a boolean value."
            return rst
        elif isinstance(trigger, list):
            return any(self._match_trigger(t, sender) for t in trigger)
        else:
            raise ValueError(f"Unsupported trigger type: {type(trigger)}")

    # def get_human_input(self, prompt: str) -> str

    # async def a_get_human_input(self, prompt: str) -> str

    # def run_code(self, code, **kwargs)

    # def execute_code_blocks(self, code_blocks)

    # def _format_json_str(jstr)

    # def execute_function(self, func_call, verbose: bool = False) -> Tuple[bool, Dict[str, str]]

    # async def a_execute_function(self, func_call)

    # def generate_init_message(self, **context) -> Union[str, Dict]

    # def register_function(self, function_map: Dict[str, Callable])

    # def update_function_signature(self, func_sig: Union[str, Dict], is_remove: None)

    # def can_execute_function(self, name: str) -> bool

    # def function_map(self) -> Dict[str, Callable]

    # def _wrap_function(self, func: F) -> F

    # def register_for_llm

    # def register_for_execution


def load_config(file_path):
    with open(file_path, "r") as f:
        config = json.load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Start an agent server.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    return parser.parse_args()


def signal_handler(sig, frame):
    print("Terminating agent")
    agent.stop()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    args = parse_args()
    config = load_config(args.config)

    config_list = config_list_from_json("../../../../OAI_CONFIG_LIST", filter_dict={"model": ["llama-v2"]})
    config_list_gpt = config_list_from_json(
        "../../../../OAI_CONFIG_LIST", filter_dict={"model": ["gpt-3.5-turbo-1106"]}
    )
    llm_config = {
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    }

    agent = StandAloneAgent(
        agent_config=config,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        llm_config=llm_config,
        human_input_mode="NEVER",
    )
    agent.serve()
    if config["name"] == "Bob":
        receiver = Agent("Anna")
        agent.send(message=f"Hello beautiful girl!", recipient=receiver, request_reply=True)
    while True:
        time.sleep(1)
