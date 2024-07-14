# key
import os
from typing import Union

from langchain.chains.llm_math.base import LLMMathChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.tools import Tool, BaseTool
from langchain_core.prompts.prompt import PromptTemplate
from Text2SQL import SQLAgent
from langchain.utilities import PythonREPL
# 配置
os.environ["TAVILY_API_KEY"] = "..." # api key
os.environ["OPENAI_API_KEY"] = "..." # api key
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
db = SQLDatabase.from_uri("...") # database path

# 设置提示模版
# prompt_o = hub.pull("hwchase17/react")
prompt = PromptTemplate(
    input_variables=['agent_scratchpad', 'input', 'chat_history', 'tool_names', 'tools'],
    template=
    """Assistant 是经过大量数据训练的大型语言模型。Assistant 
    的设计目的是能够协助完成各种任务，从回答简单的问题到提供对广泛主题的深入解释和讨论。作为语言模型，Assistant
    能够根据收到的输入生成类似人类的文本，使其能够进行听起来很自然的对话，并提供与当前主题相关且连贯的响应。Assistant 
    不断学习和改进，其功能也在不断发展。它能够处理和理解大量文本，并可以利用这些知识对各种问题提供准确和翔实的回答。此外，Assistant
    能够根据收到的输入生成自己的文本，使其能够参与讨论，并提供关于各种主题的解释和描述。总的来说，Assistant
    是一个功能强大的工具，可以帮助完成各种任务，并提供有关各种主题的有价值的见解和信息。无论是需要针对特定问题提供帮助，还是只想就某个特定话题进行对话，Assistant 
    都可以提供帮助。你的任务是回答用户向你提出的关于各个领域的问题，你非常擅长分析他们提出的问题，并且擅长使用工具和分析工具给出的答案。  
    
    你可以使用如下工具：
    {tools}
    
    回答问题时使用以下格式：

    Question: 现在要回答的问题

    Thought: 你总是要思考该如何去解决问题。

    Action: 所采取的行动，需要是下面其中之一[{tool_names}]

    Action Input:行动的输入

    Observation:行动的输出

    (...以上 Thought/Action/Action Input/Observation 的过程将重复执行数遍)

    Thought：完成了所有的步骤，这个答案符合要求，我知道最终答案了。

    Final Answer：原始问题的最终答案以及合理的解释

    开始吧！
    {chat_history}
    
    Question: {input}
    
    让我们将问题分解为多个步骤，一步一步地通过(Thought/Action/Action Input/Observation)的过程解决问题以确保最终答案是正确的。
    
    Thought: {agent_scratchpad}
"""
)

llm_prompt = PromptTemplate(
    input_variables=["input"],
    template="你是一个负责解决问题的推理工具，回答用户基于逻辑的问题。"
             "从逻辑上得出解决方案，并且这个方案是现实的。"
             "让我们一步一步地进行思考来确保答案是正确的"
             "在您的答案中，清楚地详细说明所涉及的步骤，并给出最终答案。\n"
             "如果输入中没有具体的问题，请根据输入与其进行普通对话"
             "Question: {input} \n"
             "Answer: 你的回答\n"
)
llm_math_prompt = PromptTemplate(
    input_variables=["input"],
    template="你可以熟练地生成一段Python代码来解决下面的问题, 要求在最后得到的参数使用print()函数"
             "{input},你的回答只包含代码",
)

# 设置工具
llm_chain = LLMChain(llm=llm, prompt=llm_prompt)
MathChain = LLMChain(llm=llm, prompt=llm_math_prompt)
search = TavilySearchAPIWrapper()
tavily = TavilySearchResults(api_wrapper=search)

wrapper = DuckDuckGoSearchAPIWrapper(region="de-de", time="d", max_results=2)
ddg_search = DuckDuckGoSearchRun(api_wrapper=wrapper, source="news")
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
sql_agent = SQLAgent(llm=llm, db=db)


class CalculatorTool(BaseTool):
    name = "CalculatorTool"
    description = """
    当需要回答关于数学问题或是需要运行Python程序的问题时，你可以使用这个工具，你的输入是一个数学表达式
    """

    def _run(self, question: str):
        try:
            response = MathChain.run(question)
            res = PythonREPL()
            if response[0:9] == "```python":
                output = response[9:-3]
                return res.run(output)
            else:
                return res.run(response)
        except:
            return "这个工具无法得到有效的答案，请尝试其他工具。"

    def _arun(self, value: Union[int, float]):
        raise NotImplementedError("This tool does not support async")


cal = CalculatorTool()

tools = [
    Tool(
        name='逻辑推理工具',
        func=llm_chain.run,
        description='使用该工具进行逻辑推理或是文本生成，也可以用来进行普通的对话，不使用该工具进行搜索查找'
    ),
    Tool(
        name="计算工具",
        description="当需要回答关于数学问题或是需要运行Python程序的问题时可以使用这个工具，你的输入是一个数学表达式",
        func=cal.run,
    ),
    Tool(
        name='计算器',
        description="当你需要计算一个数学表达式时可以使用这个工具，如果这个工具不能使用，请使用‘计算工具’",
        func=LLMMathChain(llm=llm).run
    ),
    Tool(
        name="维基百科",
        func=wikipedia.run,
        description="当你想知道某件事物的具体描述和背景细节时可以使用此工具，将结果翻译成中文"
    ),
    Tool(
        name='DuckDuckGo',
        description='这是一个搜索引擎工具，可以使用这个工具在线查找一些最近的信息，也可以搜索你想知道的任何信息。',
        func=ddg_search.run,
    ),
    Tool(
        name="Tavily",
        description='这是一个限制搜索数量的搜索引擎工具，如果使用DuckDuckGo搜索不到时可以使用这个工具搜索一些信息',
        func=tavily.run,
    ),
    Tool(
        name="数据库",
        func=sql_agent.run,
        description="这个工具用来执行数据库查询，当涉及到与这些表："
                    + str(db.get_usable_table_names()) + "有关的信息时，可使用此工具，如果没有查询到相关信息，请使用其他工具"
    ),
]

# initialize agent with tools
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)


# response=agent_executor.invoke({
#     "input": "那么他们的国旗长什么样？",
#     "chat_history": ["人类：加拿大的人口有多少？", "机器人：不知道呢"]
# })


def run_react_agent(input_item, llm_history):
    history = llm_history
    response = agent_executor.invoke({
        "input": input_item,
        "chat_history": history
    })
    history.append("人类：" + input_item)
    history.append("机器人：" + response["output"])
    return response['output'], history


if __name__ == '__main__':
    chat_history = []
    run_react_agent(
        "chat gpt 是什么",
        chat_history)
