import os
from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent, tool
from langchain_core.prompts import PromptTemplate
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.utilities import PythonREPL
# from prompts import QUERY_PROMPT_TMPL
from langchain_openai import ChatOpenAI
from langchain.chains import LLMMathChain
# from langchain.chains import LLMChain
from diffusers import DiffusionPipeline
import torch
# import numpy as np
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading sdlx model")
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",torch_dtype=torch.float32)
#pipe = pipe.to("cuda")
#pipe.load_lora_weights("./lora-trained-xl")

@tool
def get_word_length(word: str) -> int:
    """Returns the length of the word."""
    return len(word)

@tool    
def get_image_sdxl(text:str) -> str:
    """Generates an image from a a text-only prompt. Useful when a text is given and asked to generate an image."""
    
    image = pipe(text,num_inference_steps=2).images[0]
#    image = Image.fromarray(np.array(json.loads(response.text), dtype="uint8"))
    img_url = "./images/sdxl_output_" + "_".join(text.split()) + ".jpg"
    image.save(img_url)
    return img_url

os.environ["OPENAI_API_BASE"] = "http://localAI-service:8080/v1"
os.environ["OPENAI_API_KEY"] = "NONE"

message_history = SQLChatMessageHistory(
    session_id="test_session_id", connection_string="sqlite:///sqlite.db"
)

memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=message_history
)

# llm = OpenAI(model="mixtral", temperature=0)
chat_llm = ChatOpenAI(model="mixtral", temperature=0)

wrapper = DuckDuckGoSearchAPIWrapper(max_results=2)
search = DuckDuckGoSearchResults(api_wrapper = wrapper)
llm_math_chain = LLMMathChain.from_llm(llm=chat_llm, verbose=True)
python_repl =  PythonREPLTool()                      

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="""Use this tool in the following circumstances:
            - User is asking about current events or something that requires real-time information
            - User is asking about some term you are totally unfamiliar with (it might be new)
            - User explicitly asks you to browse or provide links to references
        """
    ),
    Tool(
        name="Math Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math, but not date calculations, only math",
    ),
    Tool(
        name="python_repl",
        func=python_repl.run,
        description="A Python shell. Use this to write and execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    ),
    Tool(
        name="Word_Length",
        func= get_word_length,
        description="useful for when you need count number of letters in a word",
    ),
    Tool(
        name="Image_Generation",
        func= get_image_sdxl,
        description="useful for when you need generate new image from text",
    )
]
                                        
template =     """Answer the following questions as best you can. You have access to the following tools:

{tools}

To use a tool, use the following format:

Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
Thought: Do I need to use a tool? No
Final Answer: [your response here]

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)
agent = create_react_agent(chat_llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

# output = await agent_executor.ainvoke({"input": "What is the LCM of 12345 and 34342"})
# output = await agent_executor.ainvoke({"input": "How many people live in india?"})
#output = await agent_executor.ainvoke({"input": "how many letters in the word Anurag?"})
#print(output)
#print(memory.load_memory_variables({}))


@app.get("/infer")
async def infer(text: str):
  
    output = agent_executor.invoke({"input": text})
    print(memory.load_memory_variables({}))
    return output

@app.get("/health")
async def health():
    return {"Health": "Ok"}



