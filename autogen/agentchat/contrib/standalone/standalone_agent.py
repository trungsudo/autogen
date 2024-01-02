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
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import requests
from central_server import AgentInfo, RunningState
from flask import Flask, request
from pydantic import BaseModel
from termcolor import colored
from werkzeug.serving import make_server

from autogen import Agent, ConversableAgent, config_list_from_json
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

        self.app = Flask(__name__)
        self.server = make_server(self._host, self._port, self.app)

        @self.app.route("/message", methods=["POST"])
        def receive_http_message():
            message = Message(**request.get_json())
            print(colored(f"{message.sender_name} -> {self.name}: {message.message}", "green"), flush=True)
            thread = threading.Thread(target=self.handler, args=(message,))
            thread.start()
            return "Message received", 200

    def handler(self, message: Message):
        self.receive(
            message=message.message,
            sender=Agent(message.sender_name),
            request_reply=message.request_reply,
            silent=message.silent,
        )

    def get_agent_from_name(self, agent_name: str) -> AgentInfo:
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

    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        valid = self._append_oai_message(message, "assistant", recipient)

        if valid:
            # recipient.receive(message, self, request_reply, silent)
            try:
                target_url = self.get_agent_from_name(recipient.name).endpoint + "/message"
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
        self.send(message, recipient, request_reply, silent)

    def receive(
        self,
        message: Union[Dict, str],
        sender: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        self._process_received_message(message, sender, silent)
        if request_reply is False or request_reply is None and self.reply_at_receive[sender] is False:
            print(colored(f"{self.name} is not gonna reply to {sender.name}", "red"), flush=True)
            return

        reply = self.generate_reply(messages=self.chat_messages[sender], sender=sender)
        if reply is not None:
            # self.send(reply, sender, silent=silent)
            try:
                target_url = self.get_agent_from_name(sender.name).endpoint
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
        self.receive(message, sender, request_reply, silent)

    def generate_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        exclude: Optional[List[Callable]] = None,
    ) -> Union[str, Dict, None]:
        print(colored(f"{self.name} generate_reply", "blue"), flush=True)
        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            logger.error(error_msg)
            raise AssertionError(error_msg)

        if messages is None:
            messages = self._oai_messages[sender]

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
        if messages is None:
            messages = self._oai_messages[sender]

        # TODO: #1143 handle token limit exceeded error
        response = client.create(
            context=messages[-1].pop("context", None), messages=self._oai_system_message + messages
        )

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
