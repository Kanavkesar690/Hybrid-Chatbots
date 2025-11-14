from langchain.callbacks.base import BaseCallbackHandler
from typing import Any

# Custom synchronous callback handler to capture results from specific tools
class MyCustomCallback(BaseCallbackHandler):
    def __init__(self):
       self.sql_result = []
       self.blob_result = []

    def on_tool_end(self, output: Any, **kwargs: Any,) -> Any:
        """Run when the tool ends running."""
        tool_name = kwargs.get("name", "")
        
        # Check which tool was used and store its result accordingly
        if tool_name == "SQLServerDBTool":
            self.sql_result = output  
        elif tool_name == "AzureBlobStorageRetrieverTool":
            self.blob_result = output  

    def get_sql_result(self):
        return self.sql_result
    
    def get_blob_result(self):
        return self.blob_result