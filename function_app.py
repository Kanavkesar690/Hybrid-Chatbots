import azure.functions as func
from HybridFunction.Hybrid_Main import bot
import logging

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Route For Bot Query
@app.route(route="HybridBotQuery")
async def BotQuery(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    query = req.params.get('query')
    if not query:
        try:
            req_body = req.get_json()
        except ValueError:
            req_body = None
        else:
            query = req_body.get('query') if req_body else None

    if query:
        response= await bot(query)
        return func.HttpResponse(response, status_code=200)
    else:
        return func.HttpResponse(
            "This HTTP triggered function executed successfully. Pass a query in the query string or in the request body for a personalized response.",
            status_code=200
        )