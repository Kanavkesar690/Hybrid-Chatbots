from langchain_openai import AzureChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain.agents import AgentType 
from azure.identity import ManagedIdentityCredential
import os 

# This class encapsulates the logic to configure an Azure-hosted LLM and build a LangChain SQL agent
class LLMDefinition:
    def __init__(self):
        self.clientid= None
        self.azure_endpoint = None
        self.openai_api_version = None
        self.deployment_name = None
        self.model = None
        self.prefix = None
        self.format_instructions = None
        self.suffix = None

    # Function to define the LLM using Azure OpenAI credentials and managed identity
    def llm_define(self):
        
        # Authenticate using Managed Identity with client ID
        credential = ManagedIdentityCredential(client_id=self.clientid)

        # Define token provider function for Azure AD authentication
        def provider():
            return credential.get_token("https://cognitiveservices.azure.com/.default").token

        # Create an instance of AzureChatOpenAI using the managed identity token
        llm = AzureChatOpenAI(
            openai_api_version=self.openai_api_version,
            temperature=0.0,
            max_completion_tokens=3000,
            # max_tokens=3000,
            azure_endpoint = self.azure_endpoint,
            openai_api_type="azure",
            deployment_name= self.deployment_name,
            model=self.model,
            azure_ad_token_provider=provider,
            
        )
        return llm
    
    # Function to create a SQL agent based on the model type and initialized LLM
    def sql_agent(self, llm):
        from HybridFunction.Hybrid_Main import db

        # Create the SQL toolkit with database and LLM
        sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        sql_toolkit.get_tools()

        # Depending on the model, create the appropriate SQL agent with specific configurations 
        if self.model.lower() in ['gpt4', 'gpt-4', 'gpt_4']:
            sqldb_agent = create_sql_agent(
                llm=llm,
                toolkit=sql_toolkit,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                agent_executor_kwargs={'handle_parsing_errors': True, 'return_intermediate_steps': False},
                verbose=True, max_execution_time=200, max_iterations=500
            )
        elif self.model.lower() in ['gpt-4o', 'gpt4o', 'gpt_4o']:
            
            # For GPT-4o, use OpenAI Functions-based agent with custom templates
            sqldb_agent = create_sql_agent(
                llm=llm,
                toolkit=sql_toolkit,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                agent_executor_kwargs={'handle_parsing_errors': True, 'return_intermediate_steps': False},
                agent_kwargs={
                    'prefix': self.prefix,
                    'format_instructions': self.format_instructions,
                    'suffix': self.suffix,
                    'input variables': ["input", "agent_scratchpad", "tool_names", "examples"],
                    # Force early stopping after the model generates an answer to avoid extra iterations
                    'early_stopping_method': None
                },
                verbose=True, max_execution_time=200, max_iterations=500
            )

        else:
            # For GPT-5, use the OpenAI_Tools agent type but limit iterations and avoid returning intermediate steps
            sqldb_agent = create_sql_agent(
                llm=llm,
                toolkit=sql_toolkit,
                agent_type="openai-tools",
                agent_executor_kwargs={'handle_parsing_errors': True, 'return_intermediate_steps': False},
                agent_kwargs={
                    'prefix': self.prefix,
                    'format_instructions': self.format_instructions,
                    'suffix': self.suffix,
                    'input variables': ["input", "agent_scratchpad", "tool_names", "examples"],

                    'early_stopping_method': None,
                },
                verbose=True, max_execution_time=200, max_iterations=500
            )
                                            
        return sqldb_agent