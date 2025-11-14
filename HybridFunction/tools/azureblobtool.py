from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain.tools import BaseTool
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
import asyncio
from typing import List, Optional, ClassVar
from ..system_instruction import SystemInstructions
from HybridFunction.Hybrid_Main import blob_tool_desc, insert_exception_log
from azure.identity import ManagedIdentityCredential

# Global variables to store Azure credential state
service_endpoint = None
key = None
indexname = None
clientid = None
resourceid = None
credentials_loaded = False

def load_azure_credentials():
    """Function to load Azure credentials only once."""
    global service_endpoint, key, indexname, clientid, resourceid, credentials_loaded
    if not credentials_loaded:
        azure_cred = SystemInstructions()
        service_endpoint, key, indexname, clientid, resourceid = azure_cred.get_azure_cred()
        credentials_loaded = True
    return service_endpoint, key, indexname, clientid, resourceid

class AzureBlobRetrieverTool(BaseTool):
    name: ClassVar[str] = "AzureBlobStorageRetrieverTool"
    description: ClassVar[str] = blob_tool_desc

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> List[dict]:
        """Asynchronous retrieval of data from Azure Blob Storage."""
        return await asyncio.to_thread(self._retrieve_data_sync, query)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[dict]:
        """Synchronous retrieval of data from Azure Blob Storage."""
        return self._retrieve_data_sync(query)

    def _retrieve_data_sync(self, query: str) -> List[dict]:
        """Synchronous retrieval helper for Azure Blob Storage queries."""
        try:
            global service_endpoint, key, indexname, clientid, resourceid
            if not service_endpoint or not key or not indexname or not clientid or not resourceid:
                service_endpoint, key, indexname, clientid, resourceid = load_azure_credentials()

            if not service_endpoint or not indexname or not clientid or not resourceid:
                return [{
                    "content": "Azure credentials are not set in the database. Please ask your admin to configure Azure settings.",
                    "source": "Not found",
                    "search_score": 0
                }]

            credential = ManagedIdentityCredential(client_id=clientid)

            search_client = SearchClient(service_endpoint, indexname, credential)

            results = search_client.search(
                search_text=query,
                select="chunk, title",
                query_type=QueryType.SIMPLE,
                include_total_count=True,
                semantic_configuration_name='semantictest-semantic-configuration',
                top=5
            )

            return self._process_results_sync(results)

        except Exception as e:
            source = 'AzureBlobRetrieverTool_run function'
            insert_exception_log(e, source)
            return [{
                "content": "Unable to find relevant data.",
                "source": "Not found",
                "search_score": 0
            }]

    def _process_results_sync(self, results) -> List[dict]:
        """Process the search results into a structured format (synchronous)."""
        response_data = []
        for result in results:
            # SearchResult objects from Azure SDK support dict-like access
            try:
                chunk = result.get('chunk')
                source = result.get('title')
                score = result.get('@search.score', 0)
            except Exception:
                # Fallback for other result shapes
                chunk = getattr(result, 'chunk', None)
                source = getattr(result, 'title', None)
                score = getattr(result, '@search.score', 0)

            if chunk and source:
                response_data.append({
                    "content": chunk,
                    "source": source,
                    "search_score": score
                })
        return response_data