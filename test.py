from fastapi import FastAPI
import uvicorn
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages, AnyMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command
from pydantic import BaseModel
import sqlite3



class TicketState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    ticket_id: str | None
    status: str | None

class ChatRequest(BaseModel):
    message: str
    thread_id: str



SYSTEM_MESSAGE = SystemMessage(content= "You are a helpful assistant, talk to the user and answer any questions")
        


@tool
def create_ticket_in_db(title: str, description: str, priority: str, user_name: str):
    """Push a completed ticket into the database."""
    print(f"DB entry created: {title}, {description}, {priority}, {user_name}")
    return True

@tool
def print_to_console(text: str):
    """Print something to the console"""
    print(f"The user said this: {text}")
    return True


class chatAPI():
    
    def __init__(self):

        self.tools_list = [create_ticket_in_db, print_to_console]
        self.llm = self._build_llm()
        self.tool_node = ToolNode(self.tools_list)
        conn = sqlite3.connect("databases/chat_history.db", check_same_thread=False)
        self.checkpointer = SqliteSaver(conn)
        self.graph = self._build_graph()

        self.app = FastAPI()
        #routes
        self.app.add_api_route("/chat", self.chat, methods=["POST"])
    
    
   
    
    def _build_graph(self):
        build = StateGraph(TicketState)
        build.add_node('model', self._model_node)
        build.add_node('tools', self.tool_node)
        build.add_edge(START, 'model')
        build.add_conditional_edges("model", tools_condition)
        build.add_edge('tools', 'model')
        graph = build.compile(checkpointer = self.checkpointer)
        return graph
    
    
    def _model_node(self, state: TicketState):
        response = self.llm.invoke(state["messages"])
        return {"messages": [response]}

    
    

    def _build_llm(self):
        return ChatOpenAI(
            model="local-model",
            base_url="http://127.0.0.1:8080",
            api_key="not-needed"
        ).bind_tools(self.tools_list)



    def chat(self, req: ChatRequest):
        config = {"configurable": {"thread_id": req.thread_id}}

        state = self.graph.get_state(config)
        is_new = not state.values

        if is_new:
            input_data = {
                "messages": [SYSTEM_MESSAGE, HumanMessage(content=req.message)]
            }
        else:
            input_data = {
                "messages": [HumanMessage(content=req.message)]
            }

        result = self.graph.invoke(input_data, config=config)

        last_message = result["messages"][-1]
        return {
            "response": last_message.content,
            "state": {
                "ticket_id": result.get("ticket_id"),
                "status": result.get("status")
            }
        }





    def run(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8000)





if __name__ == "__main__":

    api = chatAPI()
    api.run()
    
