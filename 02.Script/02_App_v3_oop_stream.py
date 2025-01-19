from Main_model_HITL_v2 import *
import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
import time
from datetime import datetime
from streamlit_cookies_manager import EncryptedCookieManager
from PIL import Image
from AWS_utils import *

image = Image.open(Path(__file__).parent.parent / '01.Data/Logo_blanco.jpeg')
st.image(image, width=150)
st.title("Agente de IA para consultas sobre el consumo de alimentos en Lima 2014-2023 analizando la Encuesta Nacional de Hogares (ENAHO)")

with st.chat_message('bot'):
    st.write("""
        ¡Hola! 👋
        Soy un agente de IA que te ayudará a responder tus preguntas sobre el consumo de alimentos en Lima 2014-2023 analizando la base de datos de la Encuesta Nacional de Hogares (ENAHO).

        Por el momento solo analizo los siguientes datos de los hogares:
        1️⃣ Nivel de pobreza
        2️⃣ Número de miembros del hogar
        3️⃣ Ingreso anual del hogar (soles)
        4️⃣ Estrato económico (A, B, C, D, E)

        Con respecto al consumo de alimentos de estos hogares:
        1️⃣ Monto anual de la compra de cada alimento (soles)
        2️⃣ Veces al año que compró el alimento
        3️⃣ Lugar en donde compró el alimento (ambulante, bodega, supermercado, etc.)

        Una habilidad importante que poseo es que puedo navegar rápidamente entre las más de 15,000 etiquetas diferentes de alimentos consumidos por los hogares y preseleccionar las que son más acordes a tu consulta. Por ejemplo, si tu consulta es sobre 'pollo a la brasa', preselecciono los alimentos relacionados como 'pollo a la brasa con papas', 'pollo broaster', etc., para que luego puedas corregir la selección.

        Una vez que ingreses tu consulta, me conecto a una base de datos de Grafos utilizando lenguaje Cypher y obtengo los datos necesarios para realizar las operaciones en un entorno de Python.

        En esta versión gratuita solo tengo acceso al 10% de la base de datos y solo puedo resolver consultas cada 4 minutos.

        Aun soy joven y puedo cometer errores, pero aprendo de ellos para ser mejor en el futuro.
             
        RECUERDA que la ENAHO es una encuesta de diseño complejo, por lo que si deseas obtener resultados que respeten ello, debes indicar explícitamente el cálculo ponderado en la pregunta.
        MUY IMPORTANTE, la calidad de mis respuestas dependerá de la calidad de tus preguntas. Por favor, sé claro y específico en tus consultas.
             
        Algunos ejemplos de preguntas que puedo responder son:
        - ¿Cuánto gastan en promedio los hogares en arroz?, calcula el promedio ponderado
        - Realiza un gráfico con la evolución anual del consumo anual en soles promedio de frutas
        - ¿Cuántas veces al año compran los hogares pollo a la brasa?
        - ¿Cuál es la distribución de la pobreza en los hogares que consumen carnes?
        

        ¿En qué puedo ayudarte hoy? 🤖💬
    """)

def _callback_dictionary():
    # Actualizar el diagnóstico en el estado
    saver =  AWSStorage()
    saver.guardar_diccionario(str(st.session_state['messages']), bucket_name = 'potatochallengelogs', s3_key_prefix = '01_App_reasoning_v7_s2/')


# Inicializar el gestor de cookies
PASSWORD = "123456APP"
cookies = EncryptedCookieManager(password=PASSWORD, prefix="my_app")
if not cookies.ready():
    st.stop()

if 'model_generator' not in st.session_state:
    model_generator = WorkflowApp()
    st.session_state['model_generator'] = model_generator
    model = model_generator._get_model() # esto no se debe ejecutar a cada rato
    st.session_state['model'] = model
    print(model.get_graph().draw_ascii())

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if "continuar_inicio" not in st.session_state:
    st.session_state['continuar_inicio'] = False

if "continuar" not in st.session_state:
    st.session_state['continuar'] = False

if "continuar_after_selection" not in st.session_state:
    st.session_state['continuar_after_selection'] = False

if "end_game" not in st.session_state:
    st.session_state['end_game'] = True

def continue_button(who):
    st.session_state[who] = True

if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

if st.session_state['end_game']:
    msg0 = st.chat_input("Escribe un mensaje")
    if msg0:
        st.session_state["user_input"] = msg0

model = st.session_state['model']
config = {"configurable": {"thread_id": "1"}}

with st.chat_message('user'):
    st.write(st.session_state["user_input"])

INTERVALO_MINIMO = 240
# Obtener la marca de tiempo actual
ahora = time.time()
# Intentar obtener la marca de tiempo de la última actualización desde la cookie
ultima_actualizacion = cookies.get('ultima_actualizacion')
if ultima_actualizacion is not None:
    ultima_actualizacion = float(ultima_actualizacion)
    tiempo_transcurrido = ahora - ultima_actualizacion
else:
    tiempo_transcurrido = INTERVALO_MINIMO + 1  # Permitir la primera actualización

try:
    if st.session_state["user_input"]:
        # esto se ejecuta solo la primera vez
        if "first_execution_done" not in st.session_state:
            #########
            if tiempo_transcurrido < INTERVALO_MINIMO:
                st.warning(f"Por favor, espera {INTERVALO_MINIMO - int(tiempo_transcurrido)} segundos antes de actualizar nuevamente.")
            else:
                # Actualizar la marca de tiempo en la cookie
                cookies['ultima_actualizacion'] = str(ahora)
                cookies.save()
                ##########
                human_message = HumanMessage(content=st.session_state["user_input"])
                # st.session_state['end_game'] = False
                input_app = {"messages": [human_message]}
                for event in model.stream(input_app, config, stream_mode="values"):
                    event['messages'][-1].pretty_print()
                    msg = event['messages'][-1]
                    print("-----------")
                    print(msg)
                    # if msg.role == "assistant":
                    if isinstance(msg, AIMessage) and msg.content!="":
                        with st.chat_message('bot'):
                            st.write(msg.content)
                    # elif isinstance(msg, ToolMessage) and msg.content!="":
                    #     with st.chat_message('bot'):
                    #         st.write(msg.content)
                    st.session_state['messages'].append(event["messages"])
                st.session_state["first_execution_done"] = True
                # if model.get_state(config).values["messages"][-1].tool_calls:
                if hasattr(model.get_state(config).values["messages"][-1], 'tool_name'):
                    tool_call_id = model.get_state(config).values["messages"][-1].tool_call_id
                    tool_call_name = model.get_state(config).values["messages"][-1].tool_name
                    st.session_state["tool_call_id"] = tool_call_id
                    st.session_state["tool_call_name"] = tool_call_name
                    print("_______________________")
                    print("WTFFFFFFFFFFFFFFFFFFFFFFFFF")
                    print("_______________________")
        else:
            if hasattr(model.get_state(config).values["messages"][-1], 'tool_name'):
                tool_call_id = st.session_state["tool_call_id"]
                tool_call_name = st.session_state["tool_call_name"]
            ################ interacción!!!! ### en cada interacción, model se modifica?
            # print("_______________________")
            # print("WTFFFFFFFFFFFFFFFFFFFFFFFFF")
            # print("_______________________")
        if (tiempo_transcurrido >= INTERVALO_MINIMO) or ("first_execution_done" in st.session_state):
            if hasattr(model.get_state(config).values["messages"][-1], 'tool_name'):
                if tool_call_name == "food_finder":
                    foods_dct, code_added_schema, added_schema, code_remove_schema = st.session_state['model_generator'].food_selector.run() # esto solo debe preseleccionar alimentos, la interacción es luego!!
                    tool_message = [
                                    RawToolMessage(
                                        ("The database schema has been updated in order to answer user's question:\n" 
                                            f"{added_schema}"),
                                        raw={'foods_correspondence': foods_dct, 'added_schema': added_schema, 'code_added_schema': code_added_schema, 'code_remove_schema': code_remove_schema},
                                        tool_call_id=tool_call_id,
                                        tool_name=tool_call_name,
                                    )]
                    # # en cada interacción, streamlit puede actualizar el valor de los alimentos, y luego de apretar el botón de continuar, se ejecuta el siguiente nodo
                    # cada cambio, modifica el estado del modelo
                    # model.update_state(config, {"messages": tool_message})#, as_node="execute_food_finder")
                    model.get_state(config).values["messages"][-1].raw['foods_dct'] = foods_dct
                    model.get_state(config).values["messages"][-1].raw['code_added_schema'] = code_added_schema
                    model.get_state(config).values["messages"][-1].raw['added_schema'] = added_schema
                    model.get_state(config).values["messages"][-1].raw['code_remove_schema'] = code_remove_schema
                    st.session_state['model'] = model
                    # ahora continuar colocando change=True
                    if st.session_state['continuar']: # si el usuario termina de seleccionar los alimentos, pero vuelve a ejecutar el nodo en el que esté
                        code_added_schema = model.get_state(config).values["messages"][-1].raw['code_added_schema'] # get code_added_schema from messages
                        print("****************")
                        print(code_added_schema)
                        print("****************")
                        for minicode in code_added_schema:
                            st.session_state['model_generator']._run_cypher_noreturn(minicode)
                        # If approved, continue the graph execution
                        for event in model.stream(None, config, stream_mode="values"):
                            event['messages'][-1].pretty_print()
                            msg = event['messages'][-1]
                            # with st.chat_message('bot'):
                            #         print("-----------")
                            #         print(msg)
                            #         st.write(msg.content)
                            if isinstance(msg, AIMessage) and msg.content!="":
                                with st.chat_message('bot'):
                                    st.write(msg.content)
        else:
            st.warning(f"Por favor, espera {INTERVALO_MINIMO - int(tiempo_transcurrido)} segundos antes de actualizar nuevamente.")
            # st.write("Historial de mensajes")
except:
    _callback_dictionary()
    st.write("Hubo un error en la ejecución del modelo")