import json
from langchain_community.callbacks import get_openai_callback
import urllib.parse
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.prompts.prompt import PromptTemplate
from dotenv import load_dotenv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .callbacks.sql_streaming import SQLHandler
from .callbacks.toolcallback import MyCustomCallback
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents.structured_chat.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
from .llm_define import LLMDefinition
from .system_instruction import SystemInstructions
import traceback
import pandas as pd
from langchain.memory import ConversationBufferMemory
from pydantic import Field
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import BaseTool 
from typing import List
from langchain.agents import AgentType
from typing import Dict, Any
from azure.ai.textanalytics import TextAnalyticsClient
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential


# This class generates a custom prompt format for LangChain agents using OpenAI function-calling mode (e.g., GPT-4o)
class LimitedConversationMemory(ConversationBufferMemory):
    max_length: int = Field(default=4) 

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context to memory and enforce max length."""
        super().save_context(inputs, outputs)
        
        if len(self.chat_memory.messages) > self.max_length:
            self.chat_memory.messages = self.chat_memory.messages[-self.max_length:]

memory = LimitedConversationMemory(memory_key="chat_history", max_length=4, return_messages=True)

load_dotenv()

query = []
api_dep = None
current_llm = None
sql_agent = None
llm = None
model = None


# SQL Server Configuration

def get_sql_connection_string():
    server = os.getenv('SERVER')
    database = os.getenv('DATABASE')
    user_name = os.getenv('USER_NAME')
    password = os.getenv('PASSWORD')
    

    conn_str = (
                r'DRIVER={ODBC Driver 17 for SQL Server};'
                f'SERVER={server};'
                f'DATABASE={database};'
                f'UID={user_name};'
                f'PWD={password};'
            )
    
    params = urllib.parse.quote_plus(conn_str)
    connection_uri = "mssql+pyodbc:///?odbc_connect=%s" % params
    
    db = SQLDatabase.from_uri(connection_uri)  
    return db 
db = get_sql_connection_string()
Suggested_Ques = db._execute("Select Question from Suggestions")
Sug_Ques = [item['Question'] for item in Suggested_Ques]

# Helper to handle exceptions including context length and SQL issues.
# This is used to log exceptions and return appropriate error messages.
def insert_exception_log(e, source):
    exception_message = str(e).replace("'", "")
    exception_type = type(e).__name__.replace("'", "")
    stack_trace = traceback.format_exc().replace("'", "")
    query = f"EXEC dbo.InsertExceptionLog N'{exception_message}', N'{exception_type}', N'{stack_trace}', N'{source}'"
    db._execute(query)

config = SystemInstructions()
config.setup_system_instructions()
System_Message = config.System_Message
task_description = config.task_description
examples = config.examples
PREFIX = config.prefix
FORMAT_INSTRUCTIONS = config.format_instructions
SUFFIX = config.suffix
hybrid_template = config.hybrid_template
sql_tool_desc = config.sql_tool
blob_tool_desc = config.blob_tool
function_description = config.function_description
ques_template = config.ques_template

from .tools.sqlserverdbtool import SQLServerDBTool
from .tools.azureblobtool import AzureBlobRetrieverTool
azure_blob_tool = AzureBlobRetrieverTool()

# Create a prompt for the LLM
def create_prompt(new_question, model_type):
    prompt_text = ""
    if model_type.lower() in ['gpt4', 'gpt-4', 'gpt_4']:
        prompt_text = f"Question: {new_question}\n"
        example_prompt = PromptTemplate(
            input_variables=["input", "query"],
            template=prompt_text,
        )
    elif model_type.lower() in ['gpt-4o', 'gpt4o', 'gpt_4o','gpt-5-mini']:  # For GPT-4o
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", new_question),
            ("ai", "{query}")
        ])

    return example_prompt

# Function to define the LLM based on the model type
def json_sql_data(handler):
    try:
        if hasattr(handler, 'sql_result') and isinstance(handler.sql_result, str):
            result = db._execute(handler.sql_result)

        elif hasattr(handler, 'sql_result') and isinstance(handler.sql_result, dict) and 'query' in handler.sql_result:
            result = db._execute(handler.sql_result['query'])

        df = pd.DataFrame(result)
        json_string = df.to_json(orient='records')
        converted_data = json_string.replace('"', '')
        converted_data2=converted_data.replace('[{','"{')
        convertedContext=converted_data2.replace('}]','}"')
        return convertedContext
    except Exception as e:
        source = 'json_sql_data function'
        insert_exception_log(e, source)
        return None

# Function to get the response in JSON format
def get_response_json(Question, response, language_code, handler, cb, sources=None, blob_content=None):
    convertedContext = None
    if handler is not None:
        if handler.sql_result is not None:
            convertedContext = json_sql_data(handler)

    suggested_questions = translate_suggested_question(Question, language_code, blob_content)

    if 'text length need to be between' in response.lower():
        response = response.split(' --> Text length need to be between')[0]

    result_with_source = {
        'source_document': sources if sources is not None else [],
        'qa_result': response,
        'total_tokens': 0,
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_cost': 0,
        'suggestions': suggested_questions,
        'StructuredData_Context':convertedContext
    }

    if cb:
        result_with_source.update({
            'total_tokens': cb.total_tokens,
            'prompt_tokens': cb.prompt_tokens,
            'completion_tokens': cb.completion_tokens,
            'total_cost': cb.total_cost,
        })

    return json.dumps(result_with_source)

# Function to get suggested questions based on cosine similarity
def get_suggested_ques(Sug_Ques, Question):
    # Combine the input question with the list of suggested questions
    questions = [Question] + Sug_Ques
    
    # Use TF-IDF vectorizer to compute similarity between the input question and suggested questions
    vectorizer = TfidfVectorizer().fit_transform(questions)
    vectors = vectorizer.toarray()
    
    # Calculate cosine similarity between the input question and the suggested questions
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    
    # Get the indices of the top 3 most similar questions
    most_similar_indices = cosine_similarities.argsort()[-4:][::-1]
    most_similar_indices = most_similar_indices[most_similar_indices != 0][:3]
    
    # Retrieve the most similar questions based on the indices
    most_similar_questions = [Sug_Ques[i-1] for i in most_similar_indices]
    
    # Prepare the list of suggested questions
    question_list = []
    for i, question in enumerate(most_similar_questions, 1):
        question_list.append(question)
    return question_list

# Function to get Azure multi-service credentials
def languagecred():  
    from HybridFunction.Hybrid_Main import db
    langDetectAndTranslateCred = db._execute("""Select Top 1 Endpoint, ApiKey from tblMultiService where [Status] = 1""")
    azure_multi_service_endpoint, azure_multi_service__key = langDetectAndTranslateCred[-1].get('Endpoint'), langDetectAndTranslateCred[-1].get('ApiKey')
    return azure_multi_service_endpoint, azure_multi_service__key

# Function to detect the language of a question
def detect_language(question):
    try:
        azure_multi_service_endpoint, azure_multi_service__key = languagecred()
        # Initialize the Azure Text Analytics client
        text_analytics_client = TextAnalyticsClient(
            endpoint=azure_multi_service_endpoint,
            credential=AzureKeyCredential(azure_multi_service__key)
        )

        # Use the client to detect the language of the input question
        response = text_analytics_client.detect_language([question])

        # Extract the detected language code and language name from the response
        detected_language = response[0].primary_language.iso6391_name
        language_name = response[0].primary_language.name
        return detected_language, language_name
    except Exception as e:
        # Log any exceptions that occur during language detection
        source = 'detect_language function'
        insert_exception_log(e, source)
        return None, str(e)

def translate_to_target(text, source_language, target_language):
    # If the source and target languages are the same, return the original text
    if source_language == target_language:
        return text
   
    try:
        azure_multi_service_endpoint, azure_multi_service__key = languagecred()
        # Initialize the Azure Text Translation client
        translation_client = TextTranslationClient(
            endpoint=azure_multi_service_endpoint,
            credential=AzureKeyCredential(azure_multi_service__key)
        )

        # Prepare the text for translation
        body = [{"text": text}]

        # Perform the translation
        response = translation_client.translate(
            body=body,
            to_language=[target_language],
            from_language=source_language
        )

        # Extract the translated text from the response
        translated_text = response[0].translations[0].text
        return translated_text
    except Exception as e:
        # Log any exceptions that occur during translation
        source = 'translate_to_target function'
        insert_exception_log(e, source)
        return str(e)

# Function to translate suggested questions to the target language
def translate_suggested_question(question, target_language, chunk_data):
    from HybridFunction.tools.SuggestionQuestionTool import CustomTools 
    try:
        if chunk_data:
            custom_tools = CustomTools(llm=llm)
            sug_ques = custom_tools.generate_suggested_prompts(
                user_question=question,
                retrieved_data=str(chunk_data.get("intermediate_steps", "")),
                agent_response=chunk_data.get("output", "")
            )
        else:
            raise ValueError("No chunk data available for generating suggestions.")
    except Exception:
        suggested_ques = db._execute("SELECT Question FROM Suggestions")
        sug_ques = [item['Question'] for item in suggested_ques]

    if not sug_ques:
        return ["", "", ""]

    # Choose top 3 suggestions depending on context
    if chunk_data:
        question_list = sug_ques[:3]
    else:
        question_list = get_suggested_ques(sug_ques, question)

    # Translate suggestions
    translated = []
    for q in question_list[:3]:
        src_lang, _ = detect_language(q)
        translated_sug_ques = translate_to_target(q, src_lang, target_language)
        translated.append((translated_sug_ques.replace('"', '')).replace("'", ""))

    return translated

# Function to make a response based on the question, final response, language code, handler, callback, and blob content
def make_response(Question, final_response, language_code, handler, cb, blob_content):
    response_text = final_response.get("response", "")
    sources = final_response.get("sources", [])

    response_language_code, _ = detect_language(response_text)
    
    translated_to_english = translate_to_target(response_text, response_language_code, 'en')
    translated_response = translate_to_target(response_text, response_language_code, language_code)

    if "ERROR:" in translated_to_english:
        if "not found" in translated_to_english.lower():
            translated_response = translate_to_target("The requested information is not available in the database.", 'en', language_code)
        else:
            translated_response = translate_to_target("Please provide more details and try again later.", 'en', language_code)

    json_data = get_response_json(Question, translated_response, language_code, handler, cb, sources, blob_content)
    
    return json_data

# Function to handle exceptions in the chatbot
def handle_chatbot_exception(e, handler, Question, language_code):    
    if hasattr(e, 'code') and (type(e).__name__.lower() == 'apiconnectionerror' or e.code.lower() == 'deploymentnotfound' or e.code == '404' or (e.status_code == 401 and type(e).__name__.lower() == 'authenticationerror')):
        response = translate_to_target('There is some issues in your Open AI Model Configuration. Please Contact your Admin', 'en', language_code)

        return get_response_json(Question, response, language_code, handler, cb=None, sources=None, blob_content=None)
    else:
        response = translate_to_target('Please provide more details and try again later', 'en', language_code)
        return get_response_json(Question, response, language_code, handler, cb=None, sources=None, blob_content=None)

# Function to initialize tools for the chatbot
def initialize_tools(handler, sql_agent, model):
    try:
        sql_tool = SQLServerDBTool(handler=handler, sql_agent=sql_agent, model=model)
        tools = [sql_tool, azure_blob_tool]
        return tools
    except Exception as e:
        source = 'initialize_tools function'
        insert_exception_log(e, source)
        return None

# Function to initialize the chatbot agent
def initialize_chatbot_agent(tools: List[BaseTool]):
    try:
        # Build a PromptTemplate from the hybrid_template and use it as the agent prompt.
        # This ensures `prompt` is defined before being passed into the agent creation.
        prompt = PromptTemplate(
            input_variables=['agent_scratchpad', 'chat_history', 'input', 'tools'],
            template=hybrid_template
        )

        agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

        # Create the executor. Keep the same options as before.
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            agent_type="openai-tools",
            handle_parsing_errors=True,
            memory=memory,
            stream_runnable=False,
            early_stopping_method="force",
        )

        return agent_executor
    except Exception as e:
        source = 'initialize_chatbot_agent function'
        insert_exception_log(e, source)
        return None

# Function to get data from the agent executor
async def get_data(question, agent_executor, handler):
    try:
        callback = MyCustomCallback()
        with get_openai_callback() as cb:
            response = await agent_executor.ainvoke(
                {
                    "input": question, 
                },
                {
                    'callbacks': [callback]
                })
        sql_response = callback.get_sql_result()
        blob_response = callback.get_blob_result()

        response = response.get('output')

        if 'final answer:' in response.lower():
            response = response.split('Final Answer:')[1].strip()

        final_response = {
               "response": response,
               "sources": []
           }
        
        sql_query = None
        if hasattr(handler, 'sql_result') and isinstance(handler.sql_result, str):
            sql_query = handler.sql_result

        elif hasattr(handler, 'sql_result') and isinstance(handler.sql_result, dict) and 'query' in handler.sql_result:
            sql_query = handler.sql_result['query']
        
        if sql_response and len(sql_response) > 0 and sql_query is not None:
               final_response["sources"].append({
                   "tool": "SQLServerDBTool",
                   "source": "SQL Database",
                   "response": sql_response,
                   "query": sql_query
               })

        blob_content = []
        if blob_response:
               for result in blob_response:  
                   if result.get("content").lower() == "azure credentials are not set in the database. please ask your admin to configure azure settings.":
                       return result.get("content"), None
                   else:
                       blob_content.append(result.get("content"))
                       final_response["sources"].append({
                       "tool": "AzureBlobStorageRetrieverTool",
                       "source": result.get("source"),
                       "content": result.get("content"),
                       "search_score": result.get('search_score')
                   })

        return final_response, cb, blob_content
    except Exception as e:
        source = 'get_data function'
        insert_exception_log(e, source)

        error_message = str(e)
        # ✅ Detect when agent stops due to max iterations
        if "Agent stopped due to max iterations" in error_message:
            friendly_message = (
                "I'm sorry, but I couldn’t complete your request as it required too many reasoning steps. "
                "Please simplify your question or try again with more specific details."
            )
            return friendly_message, None, None

        return None, None, None


# Main chatbot function that processes the question and returns a response
async def chatbot(Question, Guid, sql_agent, language_code, model):
    try:
        handler = SQLHandler()

        if 'false' not in Guid.lower().strip(' ') and model.lower() in ['gpt-4o', 'gpt4o', 'gpt_4o']:
            response = translate_to_target("Please ask your admin to turn Off Streaming.", 'en', language_code)
            query = f"EXEC dbo.InsertStreamingRecord '{Guid}', N'{Question}', N'{response}' , 1"
            db._execute(query)
            json_data = get_response_json(Question, response, language_code, handler, cb=None, sources=None, blob_content=None)
            return json_data

        tools = initialize_tools(handler, sql_agent, model)
        agent_executor = initialize_chatbot_agent(tools)

        if tools is None or agent_executor is None:
            return get_response_json(Question, "Unable to initialize agent, tools or prompt. Please contact your admin", language_code, handler, cb=None, sources=None, blob_content=None)
        
        final_response, cb, blob_content = await get_data(Question, agent_executor, handler)

        if isinstance(final_response, str):
            if final_response.lower() == "azure credentials are not set in the database. please ask your admin to configure azure settings.":
                return get_response_json(Question, final_response, language_code, handler, cb=None, sources=None, blob_content=None)

        if final_response is None or cb is None:
            return get_response_json(Question, "Unable to create a response. Please try again later", language_code, handler, cb=None, sources=None, blob_content=None)
           
        json_data = make_response(Question, final_response, language_code, handler, cb, blob_content)
        return json_data

    except Exception as e:
        source = 'chatbot function'
        insert_exception_log(e, source)
        return handle_chatbot_exception(e, handler, Question, language_code)

# Function to define the LLM and return the SQL agent, LLM, and model
def define_LLM(Question, language_code, OpenAI_Cred):
    global current_llm

    try:
        current_llm = LLMDefinition()
        current_llm.clientid = OpenAI_Cred[0]['MI_ClientID']
        current_llm.azure_endpoint = OpenAI_Cred[0]['EndPoint']
        current_llm.openai_api_version = OpenAI_Cred[0]['ApiVersion']
        current_llm.deployment_name = OpenAI_Cred[0]['DeploymentName']
        current_llm.model = OpenAI_Cred[0]['ModelName']
        current_llm.prefix = PREFIX
        current_llm.format_instructions = FORMAT_INSTRUCTIONS
        current_llm.suffix = SUFFIX

        llm = current_llm.llm_define()
        sql_agent = current_llm.sql_agent(llm)
        return sql_agent, llm, current_llm.model
    except Exception as e:
        source = 'define_LLM function'
        insert_exception_log(e, source)
        response = translate_to_target(str(e), 'en', language_code)
        return None, None, get_response_json(Question, response, language_code, handler=None, cb=None, sources=None, blob_content=None)

# Function to check for visualization requests using the global LLM
def check_for_visualization_request_with_llm(user_input):
    global current_llm

    OpenAI_Cred = db._execute(f'Select * from [ModelDetails] where id = 1')

    current_llm = LLMDefinition()
    current_llm.clientid = OpenAI_Cred[0]['MI_ClientID']
    current_llm.azure_endpoint = OpenAI_Cred[0]['EndPoint']
    current_llm.openai_api_version = OpenAI_Cred[0]['ApiVersion']
    current_llm.deployment_name = OpenAI_Cred[0]['DeploymentName']
    current_llm.model = OpenAI_Cred[0]['ModelName']

    llm = current_llm.llm_define()

    prompt = (
        "You are a classification model. Classify the following query into one of the two categories:\n"
        "1. 'Visualization Request': A request related to generating charts, graphs, tables, or any form of data visualization.\n"
        "2. 'Non-Visualization Request': Any general question or query that doesn't involve generating visualizations.\n\n"
        f"Query: {user_input}\n"
        "Response:"
    )

    try:
        response = llm.invoke(  
            input=prompt, 
            max_completion_tokens=3000,
        )
        
        # Extract and handle the response
        classification = response.content.strip()
        if classification == "Visualization Request":
            return "I am not capable to fulfill this request."
        else:
            return "Your query is being processed."
    
    except Exception as e:
        # If there is any error, log it and return an error message
        source = 'check_for_visualization_request_with_llm function'
        insert_exception_log(e, source)
        return "An error occurred while processing your request."

# Main bot function that processes the query and returns a response
async def bot(query):
    global api_dep, current_llm, sql_agent, llm, model

    response = check_for_visualization_request_with_llm(query)
    if response == "I am not capable to fulfill this request.":
        return get_response_json(query, response, language_code=None, handler=None, cb=None, sources=None, blob_content=None)

    elif 'trackstream' in query.lower():
        Question = query.split('_trackstream=')[0]
        Guid = query.split('_trackstream=')[1].split('_mId=')[0]
        Model_id = query.split('_trackstream=')[1].split('_mId=')[1]
    
    language_code, _ = detect_language(Question)
    try:
        OpenAI_Cred = db._execute(f'Select * from [ModelDetails] where id = {Model_id}')
        
        if len(OpenAI_Cred) == 1:
            new_api_dep = f"{OpenAI_Cred[0]['MI_ClientID']}_{OpenAI_Cred[0]['DeploymentName']}_{OpenAI_Cred[0]['ModelName']}"

            if api_dep is None or new_api_dep != api_dep:
                api_dep = new_api_dep
                sql_agent, llm, model = define_LLM(Question, language_code, OpenAI_Cred)

            return await chatbot(Question, Guid, sql_agent, language_code, model)
        
        elif len(OpenAI_Cred) == 0:
            response = translate_to_target('No active model is currently selected, or OpenAI credentials are not saved in the database. Please save the model and activate it before restarting your chat session.', 'en', language_code)
            return get_response_json(Question, response, language_code, handler=None, cb=None, sources=None,blob_content=None)
            
        else:
            guid_stripped = Guid.lower().strip(" ")
            current_model = current_llm.model.lower()
            is_gpt4o_model = current_model in ['gpt-4o', 'gpt4o', 'gpt_4o']
            is_guid_false = 'false' in guid_stripped

            if is_guid_false and is_gpt4o_model:
                response = translate_to_target('Multiple models are currently active. Please ensure only one model is activated at a time.', 'en', language_code)
                return get_response_json(Question, response, language_code, handler=None, cb=None, sources=None, blob_content=None)
                
            response = translate_to_target('Multiple models are currently active. Please ensure only one model is activated at a time.', 'en', language_code)
            query = f"EXEC dbo.InsertStreamingRecord '{Guid}', N'{Question}', N'{response}', 1"
            db._execute(query)
            return get_response_json(Question, response, language_code, handler=None, cb=None, sources=None, blob_content=None)
                
    except Exception as  e:
        source = 'check_LLM function'
        insert_exception_log(e, source)
        response = translate_to_target('Please contact your Network Admin regarding Open AI model configuration.','en',language_code)
        return get_response_json(Question, response,language_code, handler=None, cb=None, sources=None, blob_content=None)
    