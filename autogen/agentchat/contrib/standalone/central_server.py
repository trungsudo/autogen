from enum import Enum

from flask import Flask, request
from pydantic import BaseModel, Field


# REGISTERED -> ONLINE -> OFFLINE -> REMOVED
class RunningState(str, Enum):
    REGISTERED = "REGISTERED"
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    REMOVED = "REMOVED"


class AgentInfo(BaseModel):
    name: str = Field(..., example="Agent Smith")
    description: str = Field(..., example="Test agent")
    endpoint: str = Field(..., example="http://localhost:1232")
    status: RunningState = Field(RunningState.REMOVED, example=RunningState.REMOVED)


app = Flask(__name__)

agents = {}


@app.route("/agent", methods=["GET"])
def get_agent():
    agent_name = request.args.get("name")
    if agent_name in agents:
        return agents[agent_name].model_dump()
    else:
        return {"error": "Agent not found"}, 404


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
    if name in agents:
        print(f"REMOVED: {agents[name].name}")
        agents.pop(name)
        return {"status": "removed"}, 201
    return {"error": "Agent not found"}, 404


def clean_agent_cronjob():
    pass


if __name__ == "__main__":
    app.run(debug=False, port=8888)
