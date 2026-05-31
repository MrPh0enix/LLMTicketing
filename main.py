
from fastapi import FastAPI
import uvicorn
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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



class chatAPI():
    
    def __init__(self):

        self.model = self.get_model()
        self.structured_model = self.model.with_structured_output(LLMOutput)

        response = self.structured_model.invoke(
            "Summarize this text and determine its sentiment: I love this product."
        )
        print("This is the response: ...........................", response)
        self.chat_obj = self.get_chat_obj(self.structured_model)
        self.app = FastAPI()
        self.pending_ticket = {} 
        self.awaiting_confirmation = {}  

        #routes
        self.app.add_api_route("/chat", self.chat, methods=["POST"])

    
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
        

        response = self.chat_obj.invoke(
            {"input": req.message},
            config={
                "configurable": {
                    "session_id": session_id
                }
            }
        )


        if isinstance(response, ChatResponse):
            print('Chat response')
            print(response.message)

        if isinstance(response, ProposeTicket):
            print('Propose ticket response')
            print(response.ticket.customer_name)



        parsed = self.check_action_response_and_exec(response.content)

        if parsed is None:
            return {"response": response.content}

        # ticket proposal
        if parsed.get("action") == "confirm_ticket":

            self.pending_ticket[session_id] = parsed["ticket"]
            self.awaiting_confirmation[session_id] = True

            return {
                "response": "Please confirm this ticket (yes/no)",
                "ticket_preview": parsed["ticket"],
                "awaiting_confirmation": True
            }

        # normal chat
        if parsed.get("type") == "chat":
            return {
                "response": parsed["message"]
            }

        return {"response": response.content}


    def check_action_response_and_exec(self, response):

        try:
            data = json.loads(response)

            if data.get("type") == "propose_ticket":

                return {
                    "action": "confirm_ticket",
                    "ticket": data["ticket"]
                }

            return data

        except Exception as e:
            print("JSON parse error:", e)
            return None
            
    
    def create_ticket_in_db(self, ticket):
        print('DB entry created with: ', ticket)
    
    
    def get_model(self):
        return ChatOpenAI(model="local-model",
                          base_url="http://127.0.0.1:8080",
                          api_key="not-needed")
    

    def get_chat_obj(self, model):

        SYSTEM_PROMPT = """
                        You are a customer support assistant.
                        """
        
        prompt = ChatPromptTemplate.from_messages([
                                        ("system", SYSTEM_PROMPT),
                                        MessagesPlaceholder("history"),
                                        ("human", "{input}")
                                    ])
        
        chain = prompt | model

        return RunnableWithMessageHistory(chain,
                                    self.get_session_history,
                                    input_messages_key="input",
                                    history_messages_key="history")
    

    def get_session_history(self, session_id: str):
        return SQLChatMessageHistory(session_id = session_id,
                                     connection = "sqlite:///databases/chat_history.db")
    

    def run(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8000)





if __name__ == "__main__":

    api = chatAPI()
    api.run()
    
