from langchain.callbacks.base import BaseCallbackHandler

# Custom callback handler to intercept and capture SQL queries used by the agent
class SQLHandler(BaseCallbackHandler):
    def __init__(self):
        self.sql_result = None
 
    # This method is triggered whenever the agent decides to take an action (use a tool)
    def on_agent_action(self, action, **kwargs):
        """Run on agent action. if the tool being used is sql_db_query,
         it means we're submitting the sql and we can
         record it as the final sql"""
 
        if action.tool in ["sql_db_query", "sql_db_query_checker"]:
            self.sql_result = action.tool_input
