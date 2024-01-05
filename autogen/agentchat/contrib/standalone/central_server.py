from enum import Enum
from typing import Dict, List, Optional

from flask import Flask, jsonify, request
from pydantic import BaseModel, Field


# TODO:
# - Authorize the communication, can A talks to B?
#
# ONLINE -> OFFLINE -> UNREGISTERED
# TODO: Allow messages to be queued when OFFLINE?
class RunningState(str, Enum):
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    UNREGISTERED = "UNREGISTERED"


class AgentInfo(BaseModel):
    name: str = Field(..., example="Agent Smith")
    description: str = Field(..., example="Test agent")
    endpoint: str = Field(..., example="http://localhost:1232")
    status: RunningState = Field(RunningState.UNREGISTERED, example=RunningState.UNREGISTERED)
    # TODO: Let's Agent asks Central Server for available funcs
    functions_can_call: Optional[List[Dict]] = Field(None)
    functions_can_execute: Optional[List[str]] = Field(None)


app = Flask(__name__)

agents = {}


@app.route("/agent", methods=["GET"])
def get_agent():
    agent_name = request.args.get("name")
    if agent_name in agents:
        return agents[agent_name].model_dump()
    else:
        response = AgentInfo(
            name=agent_name,
            description="Agent is not registered",
            endpoint="None",
            status=RunningState.UNREGISTERED,
        ).model_dump()

        print(f"RESPONSE: {response}")
        return (
            response,
            404,
        )


@app.route("/register", methods=["POST"])
def register_agent():
    data = request.get_json()
    # TODO: Check existed id
    # Not new agent? => OFFLINE -> ONLINE
    if data["name"] in agents:
        return {"error": "Agent existed", "name": data["name"]}, 200

    agent = AgentInfo(**data)
    agent.status = RunningState.ONLINE
    agents[agent.name] = agent
    print(f"ONLINE: {agent.name}")
    return {"status": agent.status, "name": agent.name}


@app.route("/self_unregister", methods=["POST"])
def remove_agent():
    name = request.get_json()["name"]
    unregistered = request.get_json()["unregistered"]
    if name in agents:
        if not unregistered:
            print(f"OFFLINE: {agents[name].name}")
            agents[name].status = RunningState.OFFLINE
            return {"status": RunningState.OFFLINE}, 200
        else:
            print(f"UNREGISTED: {agents[name].name}")
            agents.pop(name)
            return {"status": RunningState.UNREGISTERED}, 200
    return {"error": "Agent not found"}, 404


def clean_agent_cronjob():
    pass


if __name__ == "__main__":
    app.run(debug=False, port=8888)
