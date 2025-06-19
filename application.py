import api 
from fastapi import FastAPI
from pydantic import BaseModel


from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

app = FastAPI()

class message(BaseModel):
    role: str
    content: str

class Messages(BaseModel):
    list_messages: list[message]
    filters: str
    history: str

def application_part(data_list, time_filter, history=""):

    q = data_list[-1].content

    graph = api.make_tool_and_workflow()

    inputs = {
        "messages": [("user",q),],
        "filters": time_filter,
        "history": history
    }

    output = graph.invoke(inputs)

    return output

@app.api_route("/remi", methods = ["GET", "POST"])
async def remi(inps: Messages):
    data_list: list[message] = inps.list_messages
    time_filter: str = inps.filters
    history: str = inps.history

    result = application_part(data_list, time_filter)

    return result



