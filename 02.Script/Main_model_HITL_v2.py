import os
import ast
import base64
import io
import json
import operator
from functools import partial
from typing import Annotated, List, Literal, Optional, Sequence, TypedDict
import pandas as pd
from IPython.display import display
from langchain_azure_dynamic_sessions import SessionsPythonREPLTool
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from matplotlib.pyplot import imshow
from PIL import Image
import neo4j
from Streamlit_selector_oop_v3 import *
#
from langgraph.checkpoint.memory import MemorySaver
from AWS_utils import *

def _callback_dictionary():
    # Actualizar el diagnóstico en el estado
    saver =  AWSStorage()
    saver.guardar_diccionario(str(st.session_state['messages']), bucket_name = 'potatochallengelogs', s3_key_prefix = '01_App_reasoning_v7_s2/')


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

class RawToolMessage(ToolMessage):
    raw: dict
    tool_name: str
    # tool_calls: list = []

@tool
class create_df_from_cypher(BaseModel):
    """
    create a df from cypher code
    """
    select_query: str = Field(..., description="A Cypher graph database SELECT statement.")
    df_columns: List[str] = Field(..., description="Ordered names to give the DataFrame columns.")
    df_name: str = Field(..., description="The name to give the DataFrame variable in downstream code.")

# @tool
class python_shell(BaseModel):
    """
    Execute Python code that analyzes the DataFrames that have been generated. Make sure to print any important results.
    The output must be a dictionary with 2 keys: code, explanation.
    """
    code: str = Field(..., description="The code to execute.")
    explanation: str = Field(..., description="The very short explanation of the code to execute.")
    
# python_shell ={
#             "name": "python_shell",
#             "description": "Execute Python code that analyzes the DataFrames that have been generated. Make sure to print any important results.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "code": {
#                         "type": "string",
#                         "description": "The code to execute. Make sure to print any important results.",
#                     },
#                 },
#             },
#         }


class food_finder(BaseModel):
    foods: List[str] = Field(..., description="The foods that are refered in the user's question")


class WorkflowApp:
    def __init__(_self,
                    AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4",
                    SESSIONS_POOL_MANAGEMENT_ENDPOINT = 'https://eastus.dynamicsessions.io/subscriptions/d87e1dc6-17dc-4498-82ba-c2d4eeb02698/resourceGroups/dynamicsession1/sessionPools/sessiones1/',
                    # SESSIONS_POOL_MANAGEMENT_ENDPOINT = 'https://eastus.dynamicsessions.io/subscriptions/7fd8277a-8a2c-4b95-b052-531e39a67f10/resourceGroups/dynamicsession1/sessionPools/sessiones1/',
                    uri = st.secrets['database']['uri'],
                    auth =  (st.secrets['database']['auth1'], st.secrets['database']['auth2']),
                    AZURE_OPENAI_API_KEY = st.secrets['interpreter']['AZURE_OPENAI_API_KEY'],
                    AZURE_OPENAI_ENDPOINT = st.secrets['interpreter']['AZURE_OPENAI_ENDPOINT']
                ):
        os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
        os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT

        # Instantiate model, DB, code interpreter
        _self.driverDB = neo4j.GraphDatabase.driver(uri, auth=auth)
        _self.llm = AzureChatOpenAI(deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME, openai_api_version="2024-02-01", temperature=0)
        _self.repl = SessionsPythonREPLTool(pool_management_endpoint=SESSIONS_POOL_MANAGEMENT_ENDPOINT)
        ######
        system_prompt = f"""
        You are an expert at Cypher graph database and Python. You have access to a Graph database in Neo4j 
        with the following schema

        {_self._get_schema()}

        The schema can be updated in the followings states of the conversation. 
        Given a user question related to the data in the database, 
        ...
        1. First, only if it is necessary, identify the foods that are refered in the user's question using the food_finder tool.
        2. Second, get the relevant data from the table as a DataFrame using the create_df_from_cypher tool.
        3. Third, Then use the python_shell to do any analysis required to answer the user question.
        """

        _self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{messages}"),
        ])
        ######
        _self.workflow = StateGraph(AgentState)
        _self.setup_workflow()

    def setup_workflow(_self):
        # _self.tools = [create_df_from_cypher, python_shell]
        # _self.tool_node = ToolNode(_self.tools)
        # _self.workflow.add_node("call_model", _self.call_model)
        # _self.workflow.add_node("execute_food_finder", _self.execute_food_finder)
        # _self.workflow.add_node("execute_cypher_query", _self.execute_cypher_query)
        # _self.workflow.add_node("execute_python", _self.execute_python)
        # _self.workflow.set_entry_point("call_model")
        # _self.workflow.add_edge("execute_food_finder", "execute_cypher_query")
        # _self.workflow.add_edge("execute_cypher_query", "execute_python")
        # # _self.workflow.add_edge("execute_food_finder", "call_model")
        # _self.workflow.add_edge("execute_python", "call_model")
        # _self.workflow.add_conditional_edges("call_model", _self.should_continue)
        #######################
        _self.tools = [create_df_from_cypher, python_shell] #, food_finder]
        # _self.tools_executors = {
        #     "create_df_from_cypher": _self.execute_cypher_query,
        #     "python_shell": _self.execute_python
        # }
        # _self.tools = [
        #     {
        #         "name": "create_df_from_cypher",
        #         "function": _self.execute_cypher_query,
        #         "description": "create a df from cypher code"
        #     },
        #     {
        #         "name": "python_shell",
        #         "function": _self.execute_python,
        #         "description": "Execute Python code in a shell."
        #     }
        # ]
        # _self.tool_node = ToolNode(tools=_self.tools)
        # _self.workflow.add_node("call_model", _self.call_model)
        # _self.workflow.add_node("execute_food_finder", _self.execute_food_finder)
        # _self.workflow.add_node("action", _self.tool_node)
        # _self.workflow.set_entry_point("call_model")
        # _self.workflow.add_edge("action", "call_model")
        # _self.workflow.add_edge("execute_food_finder", "call_model")
        # _self.workflow.add_conditional_edges("call_model", _self.should_continue,
        #                                      {"continue":'action', "execute_food_finder":'execute_food_finder', 'end':END})
        _self.workflow.add_node("call_model", _self.call_model)
        _self.workflow.add_node("execute_food_finder", _self.execute_food_finder)
        _self.workflow.add_node("execute_cypher_query", _self.execute_cypher_query)
        _self.workflow.add_node("execute_python", _self.execute_python)
        _self.workflow.set_entry_point("call_model")
        _self.workflow.add_edge("execute_cypher_query","execute_python")
        _self.workflow.add_edge("execute_python", "call_model")
        _self.workflow.add_edge("execute_food_finder", "call_model")
        _self.workflow.add_conditional_edges("call_model", _self.should_continue,
                                             {"execute_cypher_query":'execute_cypher_query',
                                              'execute_python':'execute_python',
                                              "execute_food_finder":'execute_food_finder',
                                              'end':END})

    def call_model(_self, state: AgentState) -> dict:
        messages = []
        chain = _self.prompt | _self.llm.bind_tools([create_df_from_cypher, python_shell, food_finder]) #.with_structured_output(method="json_mode")
        msg = chain.invoke({"messages": state["messages"]})
        messages.append(msg)
        print(messages)
        print("------kkkkkkkkk-----")
        return {"messages": messages}

    def execute_food_finder(_self, state: AgentState) -> dict:
        messages = []
        for tool_call in state["messages"][-1].tool_calls:
            if tool_call["name"] != "food_finder":
                continue
            foods_list = tool_call["args"]["foods"]
            _self.food_selector = FoodSelector(foods_list)
            foods_dct, code_added_schema, added_schema, code_remove_schema = _self.food_selector.initialize_variables() # esto solo debe preseleccionar alimentos, la interacción es luego!!
            # print(f"CODE: {code_added_schema}")
            # if st.session_state.continuar:
            #     for minicode in code_added_schema:
            #         _self._run_cypher_noreturn(minicode)
            messages.append(
                    RawToolMessage(
                        ("The database schema has been updated in order to answer user's question:\n" 
                            f"{added_schema}"),
                        raw={'foods_correspondence': foods_dct, 'added_schema': added_schema, 'code_added_schema': code_added_schema, 'code_remove_schema': code_remove_schema},
                        tool_call_id=tool_call["id"],
                        tool_name=tool_call["name"],
                    )
                )
        return {"messages": messages} # se interrumpe aquí after, se debe continuar con la interacción del usuario

    def execute_cypher_query(_self, state: AgentState) -> dict:
        messages = []
        print("HOLAAAAAAAAAAAAAAAAA")
        last_ai_msg = [msg for msg in state["messages"] if isinstance(msg, AIMessage)][-1]
        for tool_call in last_ai_msg.tool_calls:
            if tool_call["name"] != "create_df_from_cypher":
                continue
            df_columns = tool_call["args"]["df_columns"]
            query_str = tool_call["args"]["select_query"]
            ### revisar con chatgpt
            context = """your are an expert in Cypher language to query a database. Correct the query given by the user ONLY if it is necessary.
            Make sure that ther is no conflict between variable names, if you detect this, change the name of the variables accordingly.
            Do not change uppercase to lowercase or viceversa. Return only the query string, without preamble or conclusion and nothing more."""
            messages_alternativa = [
                {"role": "system", "content": context},
                {"role": "user", "content": query_str}
            ]
            
            # Enviar el prompt y recibir la respuesta
            pre_query_str = _self.llm(messages_alternativa)
            print(pre_query_str)
            query_str = pre_query_str.content
            ######
            df = _self.driverDB.execute_query(query_str, result_transformer_=neo4j.Result.to_df)
            print(df.head())
            df.columns = df_columns
            df_name = tool_call["args"]["df_name"]
            st.session_state[df_name] = df ### waaaa
            messages.append(
                RawToolMessage(
                    f"Generated dataframe {df_name} with columns {df_columns}",
                    # raw={df_name: df},
                    raw={df_name: df_name},
                    tool_call_id=tool_call["id"],
                    tool_name=tool_call["name"],
                )
            )
        return {"messages": messages}

    def execute_python(_self, state: AgentState) -> dict:
        print("ADIOSsssssssssssssssss")
        messages = []
        df_code = _self._upload_dfs_to_repl(state)
        last_ai_msg = [msg for msg in state["messages"] if isinstance(msg, AIMessage)][-1]
        for tool_call in last_ai_msg.tool_calls:
            if tool_call["name"] != "python_shell":
                continue
            generated_code = tool_call["args"]["code"]
            repl_result = _self.repl.execute(df_code + "\n" + generated_code)
            messages.append(
                RawToolMessage(
                    _self._repl_result_to_msg_content(repl_result),
                    raw=repl_result,
                    tool_call_id=tool_call["id"],
                    tool_name=tool_call["name"],
                )
            )
        return {"messages": messages}
        
    def should_continue(_self, state):
        # if state["messages"][-1].tool_calls:
        #     return "execute_food_finder"
        #     # return state["messages"][-1].tool_calls # esta es una lista #"execute_cypher_query" # this is the error!
        # else:
        #     code_remove_schema_list = [msg.raw['code_remove_schema']
        #                                for msg in state["messages"]
        #                                if isinstance(msg, RawToolMessage) and msg.tool_name == "food_finder"]
        #     for code_remove_schema in code_remove_schema_list:
        #         _self._run_cypher_noreturn(code_remove_schema)
        #     return END
        if not state["messages"][-1].tool_calls:
            code_remove_schema_list = [msg.raw['code_remove_schema']
                                       for msg in state["messages"]
                                       if isinstance(msg, RawToolMessage) and msg.tool_name == "food_finder"]
            print("#######")
            print(code_remove_schema_list)
            print("#######")
            for code_remove_schema in code_remove_schema_list:
                print("****************")
                print(code_remove_schema[0])
                print("****************")
                _self._run_cypher_noreturn(code_remove_schema[0])
            # st.session_state['end_game'] = False
            _callback_dictionary()
            return 'end'
        elif state["messages"][-1].tool_calls[0]["name"] == "food_finder":
            return "execute_food_finder"
        else:
            return "execute_cypher_query"
            

    def _run_cypher_noreturn(_self, cypher_code):
        with _self.driverDB.session() as session1:
            session1.run(cypher_code)
            session1.close()

    def _get_model(_self):
        memory = MemorySaver()
        return _self.workflow.compile(checkpointer=memory, interrupt_after=['execute_food_finder'])

    # def invoke(_self, input_data):
    #     app = _self.compile()
    #     return app.invoke(input_data)
    
    def _get_schema(_self):
        schema_string = """
                        # Node Properties

                ## Hogar
                - `ID_hogar`: {{type: INTEGER, label: 'ID del hogar'}}
                - `Year`: {{type: INTEGER, label: 'Año en que el hogar fue encuestado'}}
                - `Miembros_hogar`: {{type: INTEGER, label: 'Cantidad de miembros del hogar'}}
                - `Pobreza`: {{type: STRING, label: 'Nivel de pobreza del hogar', values: ['pobre extremo', 'pobre no extremo', 'no pobre']}}
                - `Importancia_hogar`: {{type: FLOAT, label: 'Ponderador de cada hogar'}}
                - `Dominio_geografico`: {{type: STRING, label: 'Dominio geográfico en el que vive el hogar', values: ['Costa Norte', 'Costa Sur', 'Sierra Norte', 'Sierra Sur', 'Selva', 'Lima Metropolitana']}}
                - `Gasto_todos_alimentos`: {{type: FLOAT, label: 'Gasto total anual del hogar en soles en todo tipo de alimentos'}}
                - `Gasto_total_hogar`: {{type: FLOAT, label: 'Gasto total anual del hogar en soles en todo'}}
                - `Ingreso_total`: {{type: FLOAT, label: 'Ingreso total anual del hogar'}}
                - `Estrato_economico`: {{type: STRING, label: 'Estrato socioeconómico del hogar', values: ['A', 'B', 'C', 'D', 'E', 'Rural']}}
                - `Ayuda`: {{type: INTEGER, label: '=1 si recibió ayuda del gobierno', values: [0, 1]}}

                ## Alimento
                - `ID_alimento`: {{type: INTEGER, label: 'ID de alimento'}}
                - `alimento_name`: {{type: STRING, label: 'Nombre del alimento'}}

                # Relationship Properties

                ## Consume
                - `Monto_compra`: {{type: FLOAT, label: 'Monto anual de compra en soles en el alimento'}}
                - `Veces_year`: {{type: FLOAT, label: 'Cantidad de veces en el año que compró el alimento'}}
                - `Lugar_compra`: {{type: STRING, label: 'Lugar en el que compra el alimento', values: [None, 'Mercado (por menor)', 'Bodega (por mayor)', 'Mercado (por mayor)', 'Bodega (por menor)', 'Supermercado', 'Ambulante (triciclo, etc.)', 'Otro', 'Feria', 'Camioneta, camión', 'Panadería', 'Restaurantes y/o bares']}}

                ## Pertenece_a
                - `(empty relationship)` 

                # The Relationships

                - `(:Hogar)-[:Consume]->(:Alimento)`
                - `(:Alimento)-[:Pertenece_a]->(:Papa_nativa)`

        """
        return schema_string
    
    def _upload_dfs_to_repl(_self, state: AgentState) -> str:
        """
        Upload generated dfs to code intepreter and return code for loading them.

        Note that code intepreter sessions are short-lived so this needs to be done
        every agent cycle, even if the dfs were previously uploaded.
        """
        df_dicts = [
            msg.raw
            for msg in state["messages"]
            if isinstance(msg, RawToolMessage) and msg.tool_name == "create_df_from_cypher"
        ]
        # name_df_map = {name: df for df_dict in df_dicts for name, df in df_dict.items()}
        name_df_map = {name: st.session_state[df] for df_dict in df_dicts for name, df in df_dict.items()}

        # Data should be uploaded as a BinaryIO.
        # Files will be uploaded to the "/mnt/data/" directory on the container.
        for name, df in name_df_map.items():
            buffer = io.StringIO()
            df.to_csv(buffer)
            buffer.seek(0)
            _self.repl.upload_file(data=buffer, remote_file_path=name + ".csv")

        # Code for loading the uploaded files.
        df_code = "import pandas as pd\n" + "\n".join(
            f"{name} = pd.read_csv('/mnt/data/{name}.csv')" for name in name_df_map
        )
        return df_code


    def _repl_result_to_msg_content(_self, repl_result: dict) -> str:
        """
        Display images with including them in tool message content.
        """
        content = {}
        for k, v in repl_result.items():
            # Any image results are returned as a dict of the form:
            # {"type": "image", "base64_data": "..."}
            if isinstance(repl_result[k], dict) and repl_result[k]["type"] == "image":
                # Decode and display image
                base64_str = repl_result[k]["base64_data"]
                img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
                st.image(img)
            else:
                content[k] = repl_result[k]
        return json.dumps(content, indent=2)

# if __name__ == "__main__":
#     app = WorkflowApp()
