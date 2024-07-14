from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os

os.environ["OPENAI_API_KEY"] = "..." # api key
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
db = SQLDatabase.from_uri("...") # database path


class SQLAgent:
    prompt = PromptTemplate(
        input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'],
        template=
        """
        你是一个专门用来解决数据库查询问题的人工智能助手，你可以将用户向你提出的问题转化为SQL，并尝试查询数据库，
        当然，你每次生成的SQL并不一定符合数据库内的实际情况，所以你可以积极尝试使用下面所提供的工具来不断纠正，
        你很有可能遇到需要跨表查询的问题，请根据获取到的数据库表信息认真发现各个表之间的联系。
    
        有以下工具可供使用：
    
        {tools}
    
        现在你已经知道你可以使用的工具了。
    
        回答问题时使用以下格式：
    
        Question：现在要回答的问题
    
        Thought：应当思考该如何去做,请分析问题，一步一步地解决问题
    
        Action：所采取的行动，可以是下面其中之一[{tool_names}]
    
        Action Input：行动的输入
    
        Observation：行动的输出
    
        (...以上 Thought/Action/Action Input/Observation 的过程将重复执行多次)
    
        Thought：我知道最终答案了。
    
        Final Answer：原始问题的最终答案
    
        以上便是你所要遵循的格式。
    
        开始吧！
    
        Question: {input}
    
        Thought: {agent_scratchpad}
    """
    )

    def __init__(self, llm, db):
        self.llm = llm
        self.db = db

    def run(self, question):
        agent_exe = create_sql_agent(
            llm=self.llm,
            toolkit=SQLDatabaseToolkit(db=self.db, llm=self.llm),
            verbose=True,
            prompt=self.prompt,
            handle_parsing_errors=True
        )
        return agent_exe.run(question)


#
if __name__ == '__main__':
    agent = SQLAgent(llm=llm, db=db)
    output = agent.run("艺术家Ed Motta的Album有哪些")
    print(output)
