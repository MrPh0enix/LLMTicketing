
from fastapi import FastAPI
import uvicorn
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel
from typing import List, Literal, Union
import json




class ChatRequest(BaseModel):
    message: str
    session_id: str


class Ticket(BaseModel):
    ticket_id: str
    customer_name: str
    issue_type: str
    severity: str
    device: str
    description: str
    steps_tried: List[str]
    status: str

class ChatResponse(BaseModel):
    type: Literal["chat"]
    message: str
    ticket_complete: bool = False

class ProposeTicket(BaseModel):
    type: Literal["propose_ticket"]
    ticket: Ticket
    ticket_complete: bool = True

class LLMOutput(BaseModel):
    content: Union[ChatResponse, ProposeTicket]


SYSTEM_PROMPT = """You are a customer support assistant."""



class chatAPI():
    
    def __init__(self):

        self.model = self.get_model()
        self.structured_model = self.model.with_structured_output(LLMOutput)

        self.checkpointer = SqliteSaver.from_conn_string("databases/chat_history.db")
        self.graph = self.build_graph()

        self.pending_ticket = {} 
        self.awaiting_confirmation = {}  

        self.app = FastAPI()
        #routes
        self.app.add_api_route("/chat", self.chat, methods=["POST"])

    
    def build_graph(self):
        def call_model(state: MessagesState):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
            response = self.structured_model.invoke(messages)
            ai_message = response

            from langchain_core.messages import AIMessage

            ai_msg = AIMessage(
                content=ai_message.content.model_dump_json() if hasattr(ai_message.content, 'model_dump_json') else str(ai_message.content),
                additional_kwargs={"structured": ai_message.model_dump()}
            )
            return {"messages": [ai_msg]}
        
        builder = StateGraph(MessagesState)
        builder.add_node("model", call_model)
        builder.add_edge(START, "model")
        builder.add_edge("model", END)
        
        return builder.compile(checkpointer=self.checkpointer)
    


    def chat(self, req: ChatRequest):

        text = req.message.lower().strip()
        session_id = req.session_id

        # confirmation flow
        if self.awaiting_confirmation.get(session_id):

            if text in ["yes", "y", "confirm", "ok"]:

                ticket = self.pending_ticket.get(session_id)

                if ticket:
                    self.create_ticket_in_db(ticket)

                    self.pending_ticket.pop(session_id, None)
                    self.awaiting_confirmation.pop(session_id, None)

                    return {
                        "response": "Ticket created successfully!",
                        "ticket": ticket 
                    }

            elif text in ["no", "n", "cancel"]:

                self.pending_ticket.pop(session_id, None)
                self.awaiting_confirmation.pop(session_id, None)

                return {
                    "response": "Okay, I cancelled the ticket."
                }

            else:
                return {
                    "response": "Please reply yes or no."
                }
            
        
        config = {"configurable": {"thread_id": session_id}}
        

        result = self.graph.invoke(
            {"messages": [HumanMessage(content=req.message)]},
            config=config
        )

        last_msg = result["messages"][-1]

        structured = last_msg.additional_kwargs.get("structured")

        if structured is None:
            return {"response": last_msg.content}
        
        content = structured.get("content", {})
        response_type = content.get("type")

        if response_type == "propose_ticket":
            ticket_data = content.get("ticket")
            self.pending_ticket[session_id] = ticket_data
            self.awaiting_confirmation[session_id] = True
            return {
                "response": "Please confirm this ticket (yes/no)",
                "ticket_preview": ticket_data,
                "awaiting_confirmation": True
            }

        if response_type == "chat":
            return {"response": content.get("message")}

        return {"response": last_msg.content}
            
    

    def create_ticket_in_db(self, ticket):
        print('DB entry created with: ', ticket)
    
    
    
    def get_model(self):
        return ChatOpenAI(model="local-model",
                          base_url="http://127.0.0.1:8080",
                          api_key="not-needed")
    

    def run(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8000)





if __name__ == "__main__":

    api = chatAPI()
    api.run()
    
