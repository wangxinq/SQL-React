# key
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from langchain_core.prompts.prompt import PromptTemplate


os.environ["OPENAI_API_KEY"] = "..." # your key
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm_prompt = PromptTemplate(
    input_variables=["input"],
    template="Question:{input}\n"
             "让我们一步步思考来确保答案是正确的"
)

llm_chain = LLMChain(llm=llm, prompt=llm_prompt, verbose=True)

response = llm_chain.run("OPenAI在什么时候发布了GPT4")
print(response)
