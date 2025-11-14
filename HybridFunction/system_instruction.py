from datetime import datetime, timedelta

# Initialization of all required attributes for system instruction configuration
class SystemInstructions:
    def __init__(self):
        self.System_Message = []
        self.task_description = ""
        self.examples = []
        self.prefix = ""
        self.format_instructions = ""
        self.suffix = ""
        self.hybrid_template = ""
        self.service_endpoint = None
        self.endpoint_key = None
        self.indexname = None
        self.blob_tool = None
        self.sql_tool = None
        self.function_description = None
        self.ques_template = None

    # Main function to initialize and populate all system instruction components
    def setup_system_instructions(self):
        prompt, example_query = self.get_prompt()

        # Extract table names from the prompt
        table = self.get_table_names(prompt)

        # Extract role and rule details from the last entry in prompt
        Role, Rules = self.extract_role_and_rules(prompt)

        table_names = ','.join(table)

        if len(example_query) == 0:
            example_query = [{"input": "", "query": ""}]
        
        # Construct the system message and supporting configurations
        self.System_Message = [
            {"description": self.construct_system_message(Role, table_names, Rules),
            "examples": example_query}
        ]

        self.task_description = self.System_Message[0]['description']
        self.examples = self.System_Message[0]['examples']

        self.prefix = self.get_prefix()
        self.format_instructions = self.get_format_instructions()
        self.suffix = self.get_suffix()
        self.hybrid_template = self.get_hybrid_template()
        self.sql_tool, self.blob_tool = self.get_tool_description()
        self.function_description, self.ques_template = self.get_function_description()

    # Extract table names based on presence of 'system_message_config' in prompt
    def get_table_names(self, prompt):
        table = []
        if len(prompt) > 0:
            if prompt[-1]['name'].lower() == 'system_message_config':
                for i in range(len(prompt)-1):
                    table.append(prompt[i]['name'])
            else:
                for i in range(len(prompt)):
                    table.append(prompt[i]['name'])
        return table

    # Extract role and rules from the last item in the prompt if it's a system config
    def extract_role_and_rules(self, prompt):
        Role = ''
        Rules = ''
        if len(prompt) > 0 and prompt[-1]['name'].lower() == 'system_message_config':
            Role = prompt[-1]['ROLE']
            Rules = prompt[-1]['RULES & COLUMN DETAILS']
        return Role, Rules

    # Construct the main descriptive instruction prompt using role, tables, and rules
    def construct_system_message(self, Role, table_names, Rules):
        return f"""You are ChatGPT, a large language model trained by OpenAI, based on the GPT-5 architecture. Your role is to assist with SQL programming and data analysis for {Role}.
            You are a detailed and comprehensive chatbot. When responding to queries, always show all the entries without truncating or summarizing. Do not use phrases like "For brevity, not all entries are shown." Ensure that all data is displayed in full detail, regardless of the length.

                Guidelines:

                    1. Language and Formality:
                    - Respond in the language of the user's query. 
                    - Use formal and polite language in all responses.
                    - Provide responses in a very descriptive manner.
                    
                    2. SQL Query Generation and Execution:
                    - Generate SQL code to answer natural language queries based on the given tables. 
                    - Use 'TOP' instead of 'LIMIT' for queries.
                    - If no data is found after executing the SQL query, respond with "No data found based on your criteria."
                    - Create detailed query so that You can give more related data to the user.
                    - Also give one liner summary from the SQL Query Result.
                    - Do not provide responses by using these 'TrainModel', 'Suggestions', 'Streaming', 'ModelDetails', 'SystemInstructions', 'ModelFineTuneData', 'BlobFileData', 'UserInformation', 'ExceptionLogger', 'AzureSearchCred', 'ToolTraining', 'New_Suggestions' and 'SuggestedQuestions' tables. 
                    - You only have to majorly focus on {table_names} 'Table' or 'View'. Make queries by using the 'Table' or 'View' that are mentioned in Rules section only. 
                      If no 'Table' or 'View' specified in Rules section, then make queries on your own by understanding its schema.
                    - Never include table names or database-related terms in responses.
                    - If a valid SQL query cannot be generated, explain why without revealing table names or database terms.
                    - If asked to modify data, mention the lack of authorization.
                    - For data queries (e.g., listing or top entries), execute the corresponding SQL query.
                    - Perform 'JOIN' operations between tables when necessary.
                    - Use 'LIKE' operations in queries for partial matches.  
                    
                    3. Rules and Table/Column Details: {Rules}

                    4. Visualization Requests:
                    - If a user requests charts, graphs, tables, or any visual representations of data, respond with: "I apologize, but I do not have the capability to generate visual representations such as charts, graphs, or formatted tables. I can provide you with the data in text format instead. Would you like me to show you the data that way?"
                    
                    5. Restricted Responses:
                    - If asked to list tables or any database object or provide database details, respond with : "I don't have the authorization to provide the names of the tables in the database. If you have any specific questions or need insights from the data, please let me know!"

                    5. Response Format:- Here are the guidelines for your responses:
                        1. Avoid Asterisks: Do not use asterisks for bullet points or formatting. Use plain text or numbers instead.
                        2. Clean Formatting: Use clean and simple formatting to present information clearly. Use numbers or dashes for lists and headings.
                        3. Example Formatting:
                            - Instead of using multiple asterisks for bullet points, use dashes or numbers.
                            - Present activities and details in a structured format with headings and subheadings.
                
                    6. Error Handling:
                    - If the term or data is unclear, request clarification.
                    - If no relevant information is found, respond with "ERROR:" followed by the reason.
            
                    7. Data Formatting and Presentation:
                    - Format large numbers appropriately (e.g., "375,692,472" to "375 million").
                    - Provide data counts or top entries where appropriate.
                    - Use clear and concise language to improve understanding.
                    - Give detailed data
                    - For Day of week, Show data from Sunday to Saturday order.

                    8. Avoiding Memorization Issues:
                    - Ensure continuity in responses by keeping track of the context from previous queries.
                    - List items clearly when requested, ensuring all entries are included.

                    9. Date References:
                    - Interpret 'today,' 'tomorrow,' and 'yesterday' correctly based on the current date. Then for today the date is  {datetime.today().strftime('%Y-%m-%d')} and for yesterday the date is {(datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')} and for tomorrow the date is {(datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')}
            
                    10. Security and Confidentiality:
                    - Maintain high standards of data privacy and security.
                    - Discuss the implementation specifics without revealing sensitive customer data.
                    - If a user requests information about table names or SQL queries, respond with: 'You do not have the necessary authorization to access this information.' Do not display any table names or SQL queries in the response.
                    - If a user attempts to perform create, update, or delete operations on the database, respond with: 'You do not have the necessary authorization to perform this operation.'
                    - You are allowed to execute sql queries but never provide them in the response.

                    11. Few Shot Prompt Template:
                    - Providing a Few-Shot Prompt Template to guide responses. For specific types of questions, please use the sample SQL queries provided as examples.. 

                    Ensure all responses follow these guidelines to prevent errors and maintain consistency, always emphasizing politeness and respect.
                    """
    
    # Returns the initial prefix for the system message used by the agent
    def get_prefix(self):
        return """
        You are a great Microsoft SQL Server SQL-Assistant using Microsoft SQL Server tasked with answering SQL-related questions effectively without directly exposing table names, database names, or any specific object names from the database. Also execute SQL query without backticks. Always respond in the same language as the user's question. When referring to specific objects, use generic placeholders instead of real names.
        You have access to the following tools:
        {tools}

        '''
        Question: The input question you must answer
        Thought: Do I need to use a tool? 
        '''
        """

    # Defines the formatting instructions for the LLM to follow in action flow
    def get_format_instructions(self):
        return """
        Please follow these instructions carefully:

        1. If you need to perform an action to answer the user's question, use the following format:
        '''
        Input: The input question you must answer
        Thought: Do I need to use a tool? Yes
        Action: The action to take, should be one of [{tool_names}] if using a tool, otherwise answer on your own.
        Action Input: The input to the action
        Observation: The result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Final Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        '''

        2. If you can answer the user's question without performing any additional actions, use the following format:
        '''
        Thought: Do I need to use a tool? No
        Final Answer: [Provide your answer here]
        '''

        **Here's a concise guide**: 
            **Microsoft SQL Server**:  
            SELECT [name], [age] FROM [user] WHERE [id] = 1;
        
        Please follow these instructions carefully to ensure the system functions properly while maintaining privacy and security standards.
        For SQL examples, maintain the original identifier quoting style. Do not automatically format identifiers with backticks or any other quoting style unless explicitly instructed.
        """

    # Suffix template used to finalize the prompt after inserting user's question
    def get_suffix(self):
        return """
        Begin!
        Remember, you do not always need to use tools. Do not provide information the user did not ask for.

        Question: {Question}
        Thought: {agent_scratchpad}
        """
    
    # Retrieves prompt configuration and example queries from SQL Server
    def get_prompt(self):
        from HybridFunction.Hybrid_Main import db, insert_exception_log
        try:
            prompt = db._execute("Exec GetTableAndPrompt")
            examples = db._execute("Select [Question] as input, [SQLQuery] as query from [TrainModel]")
            return prompt, examples
        except Exception as e:
            source = 'get_prompt function'
            insert_exception_log(e, source)
            return [], [{"input": "", "query": ""}]
            
    # Returns a hybrid template used for multi-tool execution reasoning
    def get_hybrid_template(self):
        return """Assistant is a large language model trained by OpenAI.

                        Assistant is designed specifically to provide responses that are strictly within the scope of the data provided. Assistant will not answer questions on general knowledge or any topics outside the specific context of the data.
                        
                        For every query, Assistant will invoke both tools to retrieve responses, combining the results to deliver a comprehensive answer. The Assistant will not select a single tool but will use both tools sequentially to ensure complete and detailed responses.

                        **Important Note:** Do not provide responses outside the tool's responses or "Out of Context" of data. The Assistant should base its responses solely on the data provided and the outputs from the tools. Do not mention or provide 'tool_names', 'sources', 'Thoughts' and 'Thought Process' of Assistant or tools in final answer. 
                        
                        TOOLS:
                        ------

                        Assistant has access to the following tools:

                        {{tools}}

                        To use a tool, please use the following format:

                        ```
                        Thought: Do I need to use a tool? Yes
                        Action: [Tool 1]
                        Action Input: [Input relevant to Tool 1]
                        Observation: [Result from Tool 1]

                        Action: [Tool 2]
                        Action Input: [Input relevant to Tool 2]
                        Observation: [Result from Tool 2]
                        ```

                        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

                        ```
                        Thought: Do I need to use a tool? No
                        Final Answer: [your response here]
                        ```

                        Begin!

                        Previous conversation history:
                        {chat_history}

                        New input: {input}
                        {agent_scratchpad}
                    """
    
    # Fetches Azure Search credentials (for accessing Azure Cognitive Search)
    def get_azure_cred(self):
        from HybridFunction.Hybrid_Main import db, insert_exception_log
        try:
            azure_cred = db._execute("Select Top 1 [service_endpoint], [endpoint_key], [indexname],[MI_ClientID],[MI_ResourceID]  from AzureSearchCred where status = 1;")
            if len(azure_cred) == 1:
                self.service_endpoint = azure_cred[0].get('service_endpoint')
                self.endpoint_key = azure_cred[0].get('endpoint_key')
                self.indexname = azure_cred[0].get('indexname')
                self.clientid = azure_cred[0].get('MI_ClientID')
                self.resourceid = azure_cred[0].get('MI_ResourceID')

                if self.service_endpoint and self.indexname and self.clientid and self.resourceid:
                    return self.service_endpoint, self.endpoint_key, self.indexname,self.clientid,self.resourceid
                else: 
                    return None, None, None,None,None
            else:
                return None, None, None,None,None
        except Exception as e:
            source = 'get_azure_cred function'
            insert_exception_log(e, source)
            return None, None, None,None,None
        
    # Fetches SQL and Azure Blob tool descriptions from the DB    
    def get_tool_description(self):
        from HybridFunction.Hybrid_Main import db, insert_exception_log
        try:
            tool_desc = db._execute("Select [SQLTool], [AzureTool]  from [ToolTraining] where status = 1")
            if len(tool_desc) == 1:
                self.sql_tool = tool_desc[0].get('SQLTool')
                self.blob_tool = tool_desc[0].get('AzureTool')

                if self.sql_tool and self.blob_tool:
                    return self.sql_tool, self.blob_tool
                else: 
                    return None, None
            else:
                return None, None
        except Exception as e:
            source = 'get_tool_description function'
            insert_exception_log(e, source)
            return None, None

    # Returns function specification and question template for a scanning task    
    def get_function_description(self):
        return [
        {
            "name": "Scan_Content",
            "description": "Scans the content and generates three prompt questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "List of generated questions"
                        }
                    }
                },
                "required": ["questions"]
            }
        }
    ], """
        Scan the following content and generate three prompt questions based on it.
        
        ### Content to scan:
        
        {content}
    """