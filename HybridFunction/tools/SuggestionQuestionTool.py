from langchain.tools import StructuredTool
from typing import List
import json

class CustomTools:
    def __init__(self, llm):
        """
        Initialize the CustomTools with a database instance.
        
        Args:
            llm: The language model instance to use for generating prompts.
        """
        self.llm = llm
            
    def generate_suggested_prompts(self, user_question: str, retrieved_data: str, agent_response: str) -> List[str]:
        """
        Generate 3 suggested follow-up prompts based on user question,
        SQL data, and agent response.

        If LLM cannot process, let the exception propagate to the main code.
        """
        if not self.llm:
            raise ValueError("LLM instance is required to generate suggested prompts.")

        system_message = """
        You are an assistant that suggests follow-up questions.

        Rules:
        1. If the user greets you (e.g., "hello", "hi", "hey", "good morning", "good evening", "good afternoon"),
        then your response should ONLY be conversational. 
        Example suggestions:
        - "How are you today?"
        - "Would you like me to explain what I can do for you?"
        - "Do you want to ask about hotel performance or data insights?"
        2. Base every suggestion strictly on fields, column names, or facts present in the retrieved_data or the intermediate steps. Do not invent new topics, numbers, entities, or data that are not present.
        3. Keep each suggestion short and focused (prefer 6–10 words; maximum ~12 words). Avoid long sentences.
        4. If retrieved_data is empty, unclear, or insufficient to create data-specific prompts, produce three clarifying or next-step questions that ask the user what to focus on (e.g., "Which date range should I use?", "Which metric matters most to you?", "Do you want a summary or details?").
        5. Prefer referencing column/field names exactly as they appear in the retrieved_data (e.g., use "revenue", "date", "hotel_name" if those fields exist).


        ❌ Do NOT suggest creating plots, graphs, charts, visuals, or images.
        ❌ Keep the suggestions strictly text/data-related.
        ❌ Do NOT suggest a questions to generate SQL queries.
        ❌ Do NOT reference any data, entities, or facts not present in the retrieved_data
        ✅ Keep them relevant, simple, and conversational.
        ✅ Create the suggested question only based on data you get.
        """

        prompt = f"""
        {system_message}

        The user asked: {user_question}
        The SQL Agent retrieved data: {retrieved_data}
        The Agent responded: {agent_response}

        Based on this, generate 3 useful follow-up questions the user might ask next.
        Return them as a JSON list of strings.
        """

        result = self.llm.invoke(prompt)

        # Extract content if it's an AIMessage
        result_str = getattr(result, "content", str(result)).strip()

        # Remove markdown fences
        result_str = result_str.removeprefix("```json").removesuffix("```").strip()

        # Parse JSON
        suggestions = json.loads(result_str)

        if not isinstance(suggestions, list) or len(suggestions) == 0:
            # Fallback to line-splitting
            suggestions = [line.strip("-• ") for line in result_str.split("\n") if line.strip()]

        return suggestions[:3]
    
    def get_tools(self) -> list[StructuredTool]:
        """
        Create and return a list of StructuredTool instances for use with LangChain agents.
        
        Returns:
            list[StructuredTool]: List of configured tools.
        """
        
        suggestion_tool = StructuredTool.from_function(
            func=self.generate_suggested_prompts,
            name="SuggestedPromptsTool",
            description=(
                "Generate 3 follow-up prompts based on the user question, "
                "SQL Agent retrieved data, and agent response."
            )
        )
        
        return [suggestion_tool]