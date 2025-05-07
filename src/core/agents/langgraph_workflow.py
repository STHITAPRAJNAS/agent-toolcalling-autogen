from typing import Annotated, List, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import PydanticOutputParser
from bria_lim_chatbot.agents import AgentState # Assuming AgentState is correct
from bria_lim_chatbot.service import import confluence_anthropic_prompts # Likely needs correction, maybe: from bria_lim_chatbot.service.something import ...
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import OutputFixingParser # Corrected import path
from bria_lim_chatbot.service.prompts.databricks_and_confluence_anthropic_prompts import *
from langchain_core.messages import (
    AnyMessage,
    AIMessage,
    HumanMessage,
    ChatMessage,
    SystemMessage,
    FunctionMessage,
)

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field
from loguru import logger
import pydantic
from pydantic import BaseModel, Field, model_validator
from typing import Any
import json

MAX_ATTEMPT = 3

class ChatbotAgent:

    def __init__(
        self,
        confluence_store_retriever,
        databricks_store_retriever,
        confluence_sql_connection_provider, # Might be confluence_sql_connection_provider based on usage below?
        databricks_sql_connection_provider,
        bedrock_llm_service
    ):
        self.confluence_store_retriever = confluence_store_retriever
        self.databricks_store_retriever = databricks_store_retriever

        self.llm = bedrock_llm_service.get_llm()
        # self.graph_builder = StateGraph(AgentState) # This line seems duplicated or misplaced, appears again below
        self.databricks_sql_connection_provider = databricks_sql_connection_provider
        self.history_length = 5

        self.graph_builder = StateGraph(AgentState)
        self.graph_builder.add_node("start_conversation", self.start_conversation)
        self.graph_builder.add_node("check_history", self.check_history)
        self.graph_builder.add_node("question_rewriter", self.question_rewriter)
        self.graph_builder.add_node("question_classifier", self.question_classifier)
        self.graph_builder.add_node("route_and_fetch_context", self.route_and_fetch_context)
        # self.graph_builder.add_node("fetch_confluence_context", self.fetch_confluence_context) # This line is commented out in the image
        self.graph_builder.add_node("fetch_databricks_context", self.fetch_databricks_context)

        self.graph_builder.add_node("clarify_question", self.clarify_question)
        self.graph_builder.add_node("sql_generation", self.sql_generation)
        self.graph_builder.add_node("sql_correction", self.sql_correction)
        self.graph_builder.add_node("sql_execution", self.sql_execution)
        self.graph_builder.add_node("fetch_databricks_metadata", self.fetch_databricks_metadata)
        self.graph_builder.add_node("end_conversation", self.end_conversation)

        # Define the flow of execution
        self.graph_builder.add_edge(START, "start_conversation")
        self.graph_builder.add_edge("start_conversation", "check_history")
        self.graph_builder.add_conditional_edges(
            "check_history",
            self.should_rewrite_question, # Assuming this is a method deciding the path
            {
                "rewrite_question": "question_rewriter",
                "continue": "question_classifier"
            }
        )

        self.graph_builder.add_edge("question_rewriter", "question_classifier")
        self.graph_builder.add_conditional_edges(
            "question_classifier",
            self.route_after_classifier, # Assuming this is a method deciding the path
            {
                "fetch_confluence_context": "route_and_fetch_context", # Corrected based on node name above
                "fetch_databricks_context": "route_and_fetch_context", # Corrected based on node name above
                "clarify_question": "clarify_question"
            }
        )

        # The following line seems incomplete as it likely continues off-screen
        # self.graph_builder.add_conditional_edges("fetch_databricks_context", self.route_after_fetch, {
        #     "fetch_databricks_metadata": "fetch_databricks_metadata",
        #     "sql_generation": "sql_generation",
        #     ...

#     "answer_generation": "answer_generation" # Commented out in image
        # }) # Commented out in image

        self.graph_builder.add_edge("fetch_databricks_context", "sql_generation")
        # self.graph_builder.add_edge("fetch_confluence_context", "answer_generation") # Commented out in image
        self.graph_builder.add_edge("sql_generation", "sql_correction")
        # Note: "sql_validation" node not defined in previous image snippet. Assuming it's defined elsewhere or intended differently.
        self.graph_builder.add_edge("sql_validation", "sql_execution")
        self.graph_builder.add_edge("sql_correction", "sql_execution")
        self.graph_builder.add_edge("sql_execution", "fetch_databricks_metadata")

        # self.graph_builder.add_edge("sql_generation", "answer_generation") # Commented out in image
        # self.graph_builder.add_edge("fetch_non_confluence_content", "answer_generation") # Commented out in image

        # self.graph_builder.add_edge("answer_generation", END) # Commented out in image
        self.graph_builder.add_edge("clarify_question", "end_conversation") # Connects clarification directly to end
        self.graph_builder.add_edge("fetch_databricks_metadata", "end_conversation")

        self.graph_builder.add_edge("end_conversation", END)

        # Assuming 'checkpointer' is defined and passed appropriately elsewhere
        self.chatbot_graph = self.graph_builder.compile(checkpointer=checkpointer)


    def llm_invoke_and_get_content(self, prompt):
        response = self.llm.invoke(prompt)
        # llm_response_content = [i['text'] for i in response.content if i['type']=='text'] # Commented out in image
        # Assumes response.content is a list containing a dict with a 'text' key at index 0
        return response.content[0]['text']


    def start_conversation(self, state: AgentState) -> AgentState:
        logger.info("Entering start_conversation stage")
        # The next two lines seem contradictory, the second overwrites the first immediately. Transcribing as seen.
        state["messages"] = state.get("messages", [])[:]
        state["messages"] = []
        # Append human message
        # Assuming state["human_message_content"] holds the user's input
        logger.debug(f'Appended human message: {state["human_message_content"]}')
        state["messages"].append(HumanMessage(content=state["human_message_content"]))
        return state


    def end_conversation(self, state: AgentState) -> AgentState:
        logger.info("Entering end_conversation stage")
        # Assuming state["final_answer"] holds the generated answer
        ai_message = AIMessage(content=state["final_answer"], name="Tesseract") # Assigns name "Tesseract"
        state["messages"].append(ai_message)
        logger.debug(f'Appended AI message: {ai_message.content}')
        return state


    def check_history(self, state: AgentState) -> AgentState:
        logger.info("Entering check_history stage")
        # Initialize state variables for the current run
        state["intermediate_steps"] = []
        state["databricks_context"] = []
        state["generated_query"] = ""
        state["sql_error"] = ""
        state["needs_clarification"] = False
        state["clarification_response"] = ""
        state["final_answer"] = ""
        # The following 'else' seems out of place without a preceding 'if' in this snippet
        # # Key Conversation name not in state
        # else state['conversation_name'] = ""

        state["intermediate_answers"] = [] # Initialize list for intermediate answers
        history = state["messages"] # Get current messages

        # Check if messages exist
        if not state["messages"] or state["messages"] is None:
            state["messages"] = []

        # Check if there's history to analyze (more than just the initial user message)
        if len(state["messages"]) > 1:
            question = state["messages"][-1].content # Get the latest question
            history = state["messages"][:-1] # Get all messages except the last one

            # Assuming HISTORY_LIMIT is defined elsewhere
            history = self.get_limited_history(history, HISTORY_LIMIT)

            # Assuming check_history_template is a PromptTemplate defined elsewhere
            check_history_prompt = check_history_template.format(
                history=history, question=question
            )

            # response = self.llm_invoke_and_get_content(check_history_prompt) # This call is commented out in the image
            response = 'response string' # Placeholder value seen in image

            if 'Answer not found in history' in response:
                logger.info("Answer not found in history.")
                # self.add_intermediate_answer(state, 'History check: Answer not found') # Commented out

# Assuming check_history_prompt was defined in the previous snippet
            response = self.llm_invoke_and_get_content(check_history_prompt)
            answer = response.strip()

            # Check the response content to determine if an answer was found in history
            if "answer found:" in answer: # Specific string check
                state["history_answer"] = answer # Store the found answer
                # state['final_answer'] = answer # This line is commented out in the image
                logger.info("Answer found in history.")
            elif "Answer not found in history." in answer: # Specific string check
                state["history_answer"] = None
                logger.info("Answer not found in history.")
            else:
                # Fallback or alternative check if the initial response wasn't clear
                # Assuming check_history_answer_template is defined elsewhere
                history_answer_prompt = check_history_answer_template.format(
                    history=history, question=question
                )
                # response = self.llm_invoke_and_get_content(history_answer_prompt) # This call is commented out in the image
                # Which 'response' is being stripped here is ambiguous - assuming it's from the earlier call
                answer = response.strip()
                if 'Final Answer:' in answer: # Check for "Final Answer:" prefix
                    state["history_answer"] = answer
                    logger.info("Final answer found in history.")
                else:
                    state["history_answer"] = None
                    logger.info("No answer found in history.")

        # Check or generate conversation name if not already set
        if not state["conversation_name"]:
            # state["conversation_name"] = "New Conversation" # This line is commented out
            # Assuming check_history_conversation_name_template is defined elsewhere
            check_history_conversation_name_prompt = (
                check_history_conversation_name_template.format(history=history)
            )
            # Call LLM to generate a conversation name based on history
            response = self.llm_invoke_and_get_content(
                check_history_conversation_name_prompt)
            answer = response.strip()
            state["conversation_name"] = answer

        return state


    def question_rewriter(self, state: AgentState) -> AgentState:
        # Check if message history is less than a certain limit minus 1
        # Assuming self.HISTORY_LIMIT is defined in __init__ or elsewhere
        if len(state["messages"]) < (self.HISTORY_LIMIT -1 ):
            current_question = state["messages"][-1].content
            # Assign current question directly if history is short
            state["rephrased_question"] = current_question # Corrected key name from potential typo 'replrased'
        else:
            # Define system prompt for rephrasing
            system_prompt = """You are a helpful assistant that rephrases the user's question to be a standalone question optimized for retrieval.
Do not try to expand unknown acronyms. If there are pronouns in the question please use the conversation history to get context.
Make sure you return the original question if it cannot be rephrased."""

            # Prepare messages for the prompt template
            messages = (
                SystemMessage(content=system_prompt),
                state["messages"][-1] # Include the last message object
            )
            # Create prompt template from messages
            prompt = ChatPromptTemplate.from_messages(messages)
            # Invoke LLM to get the rephrased question
            rephrased_question = self.llm_invoke_and_get_content(prompt)
            # Store the stripped rephrased question in state
            state["rephrased_question"] = rephrased_question.strip()

        logger.debug(f"Rephrased query: {state['rephrased_question']}")
        return state


    def question_classifier(self, state: AgentState) -> AgentState:
        logger.info("Entering question_classifier stage")
        # Classify the user's question to determine intent.
        question = state["rephrased_question"]
        # Assuming question_classifier_prompt is defined elsewhere
        prompt = question_classifier_prompt.format(question=question)

        # response = self.llm_invoke_and_get_content(prompt) # This call is commented out
        intent = response # Assigns the 'response' from earlier in the code block? Needs clarification.

        state["intent"] = intent.strip()
        logger.debug(f"Intent: {state['intent']}")
        return state


    def fetch_confluence_context(self, state: AgentState) -> AgentState:
        logger.info("Fetching context from Confluence.")
        query = state["rephrased_question"]
        # results = self.confluence_retriever.invoke(query) # This call is commented out
        # Use confluence metadata api to get results
        # state["confluence_context"] = results # This assignment is commented out
        # If results is empty
        # results = self.confluence_metadata.get("source", []) # This assignment is commented out

        # --- Code potentially missing or commented out ---

# return state # End of fetch_confluence_context from previous snippet

    # Note: The definition header below seems like a duplicate and is commented out in the image.
    # def fetch_confluence_context(self, state: AgentState) -> AgentState:
    #     logger.info("Fetching context from Confluence.")
        query = state["rephrased_question"]
        # state["confluence_context"] = [] # This line is commented out
        # Assuming self.confluence_retriever is defined in __init__
        results = self.confluence_retriever.invoke(query)
        state["confluence_context"] = results
        # state["confluence_context"] = [] # This line is commented out
        # if not results: # This block is commented out
            # results = self.confluence_metadata.get("source", [])
        logger.info(f"Confluence context retrieved: {state['confluence_context']}")
        return state


    def fetch_databricks_context(self, state: AgentState) -> AgentState:
        logger.info("Fetching context from Databricks.")
        query = state["rephrased_question"]
        state["databricks_context"] = [] # Initialize context list
        # Assuming self.databricks_retriever is defined in __init__
        results = self.databricks_retriever.invoke(query)
        state["databricks_context"] = results
        # if not results: # This block is commented out
            # results = self.databricks_metadata.get("source", [])
        logger.info(f"Databricks Context retrieved: {state['databricks_context']}")
        return state


    def clarify_question(self, state: AgentState) -> AgentState:
        logger.info("Entering clarify_question stage")
        state["final_answer"] = "" # Initialize final_answer for this path
        # Get recent conversation history based on HISTORY_LIMIT
        conversation = state["messages"][-self.HISTORY_LIMIT:]
        question = state["rephrased_question"]
        # Assuming clarify_question_template is defined elsewhere
        clarify_question_prompt = clarify_question_template.format(
            conversation=conversation, question=question
        )
        # Invoke LLM to generate a clarification question
        response = self.llm_invoke_and_get_content(clarify_question_prompt)
        state["final_answer"] = response # Store the clarification as the final answer for now
        logger.info(f"Clarification question: {state['final_answer']}")
        # Setting needs_clarification to False here seems counter-intuitive,
        # perhaps it should be set True earlier when this path is chosen. Transcribing as seen.
        state["needs_clarification"] = False
        return state


    class SQLGenerationOutput(BaseModel):
        """Pydantic model representing the output of the SQL generation step."""
        thought: str = Field(description="Thought process leading to the SQL query and thought") # Typo? "thought" repeated
        description: str = Field(description="Chain of thought leading to the SQL query and generation.") # Seems similar to 'thought'
        generated_query: str = Field(description="Generated SQL query only, can be empty if no query is applicable.")


    def sql_generation(self, state: AgentState) -> AgentState:
        logger.info("Entering sql_generation stage")
        # Define the SQL generation prompt template
        # The line below seems incomplete or incorrectly formatted in the image
        # prompt = sql_generation_prompt_template # self.llm_with_structured_output(SQLGenerationOutput)
        # Assuming 'chain' and the prompt template are defined and properly used elsewhere
        # The .invoke call seems incomplete and references unclear variables/state structure
        # response = chain.invoke(...) # Incomplete call
        # metadata, context = state["metadata"], state["databricks_context"] # Assumes 'metadata' key exists
        # question = state["rephrased_question"], # Trailing comma?
        # "validation_response": state["sql"]["error"], # Assumes state['sql'] structure exists
        # validation_response=state["sql"]["validation_response"]) # Duplicate validation response?

        # Placeholder for actual response object based on SQLGenerationOutput
        response: SQLGenerationOutput = None # Replace with actual call result

        if response:
            logger.info(f"SQL thought: {response.thought}")
            # Assumes state['sql'] is a dictionary initialized elsewhere
            state["sql"]["generated_query"] = response.generated_query
            state["sql"]["thought"] = response.thought
        else:
            # Handle case where response is None or invalid
            state["sql"]["error"] = "Failed to generate SQL query"
            state["sql"]["thought"] = ""
            state["sql"]["validation_response"] = "" # Clear validation response on generation failure
        return state


    class ValidationOutput(BaseModel):
        """Pydantic model for SQL validation output."""
        is_valid: bool = Field(
            description="Evaluate the SQL query and determine if it is valid and likely to answer the user's question. Return True else False."
        )
        explanation: str = Field(
            description="Provide a brief explanation for the validation decision."
        )


# Assuming the sql_validation method definition starts here or continues from previous snippet
    def sql_validation(self, state: AgentState) -> AgentState:
        logger.info("Entering sql_validation stage") # Potentially duplicate log if method continued
        # Note: In the previous snippet, metadata seemed to be derived differently.
        # Here it's assigned directly from databricks_context. Clarification needed.
        metadata = state["databricks_context"]
        # Assuming sql_validation_prompt_template is defined elsewhere
        # Assuming self.llm_with_structured_output(ValidationOutput) is correctly set up if not using 'chain'
        chain = sql_validation_prompt_template # Or use self.llm_with_structured_output(...)
        response = chain.invoke(
            {
                "metadata": metadata,
                "question": state["rephrased_question"],
                "generated_sql": state["sql"]["generated_query"] # Assumes state['sql'] structure
            }
        )

        # Assuming response is an instance of ValidationOutput Pydantic model
        logger.info(f"SQL validation response: {response.explanation}")
        state["sql"]["validation_response"] = response.explanation
        state["sql"]["is_valid"] = response.is_valid
        logger.info(f"SQL validation result: {state['sql']['is_valid']}")
        return state


    def sql_execution(self, state: AgentState) -> AgentState:
        logger.info("Entering sql_execution stage")
        state["sql_result"] = "" # Initialize SQL result in state
        # Assuming MAX_ATTEMPT is defined globally or in __init__
        # MAX_ATTEMPTS = 3
        attempt_count = state["sql"].get("attempt_count", 0)

        # Check if max attempts reached
        if attempt_count >= MAX_ATTEMPT:
            # Construct error message (ensure f-string formatting is correct)
            error_message = (
                f"SQL execution failed: Unable to generate a valid SQL query or retrieve meaningful results "
                f"from Databricks for your question after {attempt_count} attempts."
            )
            state["sql"]["error"] = error_message
            logger.info("SQL execution abandoned after max retries")
            # state["sql"]["attempt_count"] = MAX_ATTEMPT # Comment suggests additional logic might be intended
            state["sql"]["generated_query"] = "" # Clear query after max attempts
            return state

        # Check if SQL was marked invalid in the previous step or if there's already an error
        if not state["sql"]["is_valid"] or state["sql"]["error"]:
            logger.info("Skipping SQL execution due to invalid SQL or prior generation error.")
            # Increment attempt count for the generation/validation cycle that failed
            state["sql"]["attempt_count"] = attempt_count + 1
            return state

        # If SQL is valid and no prior error, proceed with execution
        sql_query = state["sql"]["generated_query"]
        try:
            # Assuming self.databricks_sql_connection_provider is set up in __init__
            # result = self.databricks_sql_connection_provider.query_db(sql_query) # Actual call
            result = None # Placeholder for the actual result
            logger.info(f"SQL result: {result}")
            state["sql"]["result"] = str(result) # Store result as string
        except Exception as e:
            logger.error(f"SQL execution Failed: {e}")
            state["sql"]["error"] = str(e)
            # Increment attempt count on execution error to trigger retry/correction
            state["sql"]["attempt_count"] = attempt_count + 1
            logger.info(
                f"Retrying SQL generation due to execution error (attempt {attempt_count + 1})"
            )
        return state


    class FinalAnswerGenerationOutput(BaseModel):
        """Final answer generation model"""
        thought: str = Field(description="Chain of thought leading to providing the final answer, this should not cause any parsing error.")
        final_answer: str = Field(
            description="Generated final response that should be free from any parsing error that may have occurred."
        )

        # The method below seems intended as a validator or utility function,
        # but its definition structure (nested async functions) and use of 'value'
        # without definition is unclear in this context. Transcribing as seen.
        # # handle validation method
        # def _check_serialization(mode="before"):
        #     def _wrapper(func):
        #         async def _inner(self, *args, **kwargs) -> Any:
        #             try:
        #                 # Attempt to parse the JSON
        #                 json.loads(value) # 'value' is undefined here
        #                 return value
        #             except Exception:
        #                 # Handle the error by returning the original value or a default value
        #                 return value
        #                 # return default_value
        #     # Needs to return _wrapper or _inner depending on intent


    def answer_generation(self, state: AgentState) -> AgentState:
        logger.info("Entering answer_generation stage")
        context = ""
        # Combine context from Confluence if available
        if state["confluence_context"]:
            context += "\n## Context from Confluence Documents:\n"
            # Assuming confluence_context is a list of strings or objects convertible to strings
            context += "\n".join(map(str, state["confluence_context"]))

        # Combine context from Databricks if available
        # Note: This looks like it should be 'elif' or a separate 'if', not nested. Transcribing as seen.
        if state["databricks_context"]: # Might need to be elif or separate if?
            context += "\n## Databricks Table Schema Metadata:\n"
            # Assuming databricks_context is a list of strings or objects convertible to strings
            context += "\n".join(map(str, state["databricks_context"]))

# context += "\n".join(map(str, state["databricks_context"])) # Duplicate line? Commented out.

        # Add SQL query and result to context if they exist
        if state.get("sql", {}).get("generated_query") and state.get("sql", {}).get("result"):
            context += "\n## Generated SQL Query:\n"
            # Using f-string with triple quotes for potential multi-line SQL
            context += f'```sql\n{state["sql"]["generated_query"]}\n```'
            context += "\n## SQL Result:\n"
            # Using f-string with triple quotes for potential multi-line results
            context += f'```\n{state["sql"]["result"]}\n```'

        # Convert intermediate_answers list to string and add to context
        # Assuming intermediate_answers is a list of strings
        if state.get("intermediate_answers"):
             context += "\n## Intermediate Answers:\n"
             context += "\n".join(map(str, state["intermediate_answers"]))

        # Get limited history
        # Assuming self.HISTORY_LIMIT is defined
        history = self.get_limited_history(state["messages"], limit=self.HISTORY_LIMIT)
        # filter out message place holder - Note: No placeholder filtering logic is visible here
        # history = [m for m in history] # This line doesn't perform any filtering as written

        logger.debug(
            f"Generating answer with context:\n{context}\nand history:\n{history}"
        )

        # Format prompt using the combined context and history
        # Assuming answer_template is defined elsewhere
        prompt = answer_template.format(history=history, query=state["rephrased_question"], context=context)
        # Assuming self.llm_with_structured_output(FinalAnswerGenerationOutput) is set up if not using 'chain'
        # The 'chain =' line below seems incorrect; likely only the prompt or an LLM call is needed.
        chain = answer_template # Or use self.llm_with_structured_output(...)
        # Assuming the invoke method takes a dictionary corresponding to the template variables
        response = chain.invoke(
             {
                 "history": history,
                 "query": state["rephrased_question"],
                 "context": context
             }
        )

        # Assuming response is an instance of FinalAnswerGenerationOutput
        logger.info(f"Thoughts leading to final answer: {response.thought}")
        state["final_answer"] = response.final_answer.strip()
        logger.info(f"Generated answer: {state['final_answer']}")
        return state


    # Defining Routing Functions
    def should_rewrite_question(self, state: AgentState) -> str:
        # Note: This logic seems reversed. It returns "rewrite_question" if an answer *is* found.
        # Transcribing as seen, but this might need review based on intended flow.
        return "rewrite_question" if state.get("history_answer") else "question_rewriter"


    def route_after_classifier(self, state: AgentState) -> str:
        intent = state.get("intent", "").lower() # Get intent safely and lowercase it
        if "confluence" in intent:
            return "fetch_confluence_context"
        elif "databricks" in intent:
            return "fetch_databricks_context"
        else: # Default route if intent doesn't match known sources
            return "clarify_question"


    def route_after_fetch(self, state: AgentState) -> str:
        # Note: Routing logic here might need review.
        # Classifier usually directs to only one fetch_* node. Checking for both seems unusual.
        # Also, the key checked 'fetch_confluence_context' is a node name, not usually a state key.
        # Assuming intent was to check if context exists instead.
        # if state.get("confluence_context") and state.get("databricks_context"):
        if state.get("databricks_context"): # More likely check: if databricks context was fetched (for SQL)
             return "sql_generation"
        else: # Otherwise (e.g., only confluence context fetched), generate answer
             return "answer_generation"


    def route_after_sql_generation(self, state: AgentState) -> str:
        attempt_count = state.get("sql", {}).get("attempt_count", 0)
        # Check for max attempts or if intent wasn't suitable for SQL (intent check seems odd here)
        if attempt_count >= MAX_ATTEMPT: # Removed intent check as it seems misplaced
             # If max attempts reached, go to answer generation (likely with error message)
             return "answer_generation"
        # elif state['sql']["error"] or not state['sql']["is_valid"] and state['sql']: # Original boolean logic might be ambiguous
        # Check for error OR if SQL is not valid (ensuring 'sql' dict exists)
        elif state.get("sql", {}).get("error") or not state.get("sql", {}).get("is_valid", True): # Check error or invalidity
             return "sql_correction" # Route to correction if error occurred or SQL is invalid
        else: # If SQL generated without error and is valid (or validation skipped)
             return "sql_validation" # Route to validation (or execution if validation happens before generation?) Logic needs review.


    def get_limited_history(self, messages, limit):
        """Get the last 'limit' number of messages from the conversation history."""
        messages = list(messages) # Ensure it's a list copy
        # Get the last 'limit' messages. Note: Placeholder filter mentioned in comment isn't implemented here.
        # Note: Multiplying limit by 2 seems arbitrary without explanation. Using original limit.
        # limit = limit * 2 # Why multiply by 2? Reverting to standard limit usage.
        return messages[-limit:] # Return the actual last 'limit' messages


    # This function seems defined but not used in the transcribed snippets.
    # def list_to_string(self, messages: list) -> str:
    #     """The limited conversation history as a string."""
    #     history_string = ""
    #     # Assumes messages is a list of objects with 'type' and 'content' attributes or similar structure
    #     for msg in messages: # Iterate through message objects directly
    #         # Example: Adjust based on actual message object structure (e.g., AIMessage, HumanMessage)
    #         msg_type = type(msg).__name__
    #         msg_content = getattr(msg, 'content', str(msg))
    #         history_string += f"\n{msg_type}: {msg_content}"
    #     return history_string

# Note: The following routing logic might be duplicated or modified from previous snippets.

    # def route_after_classifier(self, state: AgentState) -> str: # Example partial repeat
    #     ... (logic as seen before) ...
    #     elif "databricks" in intent: # Repeated section from route_after_classifier
    #        return "fetch_databricks_context"
    #     else:
    #          return "clarify_question"

    def route_after_fetch(self, state: AgentState) -> str:
        # Updated logic: Check if databricks_context exists and has more than 1 item
        if state.get("databricks_context") and len(state.get("databricks_context", [])) > 1:
            return "sql_generation"
        else:
            return "answer_generation"

    # Note: Function name changed from route_after_sql_generation to route_after_sql_execution
    def route_after_sql_execution(self, state: AgentState) -> str:
        # Updated logic: High attempt count check (MAX_ATTEMPT + 10) and unusual 'both' intent check
        # The 'both' intent check might be specific to a use case or a potential typo.
        if state.get("sql", {}).get("attempt_count", 0) >= MAX_ATTEMPT + 10 and state.get("intent") != 'both':
            # Route differently if many attempts failed (e.g., try Confluence context?)
            return "fetch_confluence_context"
        # Check for error OR invalid SQL (ensuring 'sql' dict exists)
        elif state.get("sql", {}).get("error") or not state.get("sql", {}).get("is_valid", True):
             # Route back to sql_generation for correction (different from previous snippet's route_after_sql_generation)
             return "sql_generation"
        else: # If execution was successful (or no error/invalidity detected)
             return "answer_generation"

    # Note: Repeated definition with default limit and different return type annotation in docstring
    def get_limited_history(self, messages, limit=5):
        """Get the last 'limit' number of messages from the conversation history.

        Args:
            messages (list): List of messages containing the conversation history.
            limit (int): The number of messages to include in the history.

        Returns:
            str: The limited conversation history as a string. # Note: Docstring says str, code returns list slice.
        """
        limited_messages = messages[-limit:] # Get the last 'limit' messages
        # Example string conversion (commented out in image):
        # history = '\n'.join([f'{getattr(msg, "type", type(msg).__name__)}: {getattr(msg, "content", "")}' for msg in limited_messages])
        # The function currently returns the list slice, not the string 'history' variable mentioned in the docstring/comment.
        return limited_messages # Returning the list slice based on code logic


    def invoke_chat(self, user_input, config):
        """Invokes the chatbot graph with user input and configuration."""
        # Assuming self.chatbot_graph is the compiled LangGraph state machine
        result = self.chatbot_graph.invoke(
            {"question": HumanMessage(content=user_input)}, # Wrapping input in HumanMessage
             config # Pass config, likely including {"configurable": {"thread_id": ...}}
        )
        # return result # Commented out in image
        return result # Returns the final state or result of the graph invocation


    def get_history(self, conversation_id: str):
        """Retrieves the message history for a given conversation ID."""
        config = {"configurable": {"thread_id": conversation_id}}
        # Get the state associated with the conversation thread ID
        chat_hist_messages = self.chatbot_graph.get_state(config).values.get(
            "messages", [] # Safely get the 'messages' list from the state values
        )
        # Format the history messages for return
        return [{"message_type": message.type, "message_content": message.content}
                for message in chat_hist_messages]


    async def invoke_chat_stream(self, user_input, config):
        """Invokes the chatbot graph asynchronously and streams responses."""
        # Add user message to state manually - Note: This comment suggests state manipulation might be needed before streaming, but isn't shown here.
        # Use astream for asynchronous streaming invocation
        # Input structure seems slightly off, likely should be {"question": HumanMessage(...)} directly
        async for chunk in self.chatbot_graph.astream(
            # {"input": "question": HumanMessage(content=user_input)}, # Original structure in image
            {"question": HumanMessage(content=user_input)}, # Corrected input structure
             config,
             stream_mode="messages" # Stream individual messages/chunks
        ):
            # Process each chunk received from the stream
            chunk_message_content = chunk.content
            print(f"Receiving new event of type: {type(chunk)}: {chunk}") # Debug print
            print(f"{type(chunk)}") # Debug print type

            if chunk: # Check if the chunk object is valid
                # chunk_message_content # This line has no effect
                # Check if the chunk originates from the "answer_generation" node
                # Note: 'metadata' is not defined here; assuming it's part of the 'chunk' object's structure
                # Example: if chunk.metadata.get('langgraph_node') == "answer_generation":
                # And if the chunk is an AIMessageChunk (part of an AI response)
                if isinstance(chunk, AIMessageChunk): # Simplified check assuming AIMessageChunk implies answer generation for streaming
                    chunk_message = chunk # Assign for clarity
                    print(f"Chunk Message: {chunk_message.content}") # Print AI chunk content
                    print(f"{type(chunk)}") # Debug print type
                    # 'stream_text' is not defined; likely intended to be chunk_message.content
                    # print(f"{stream_text}")
                    stream_text = chunk_message.content
                    # Yield data in Server-Sent Event format
                    yield f"data: {stream_text}\n\n"


    def get_conversation_name(self, conversation_id: str):
        """Retrieves the conversation name for a given conversation ID."""
        config = {"configurable": {"thread_id": conversation_id}}
        # Get the state values for the conversation thread
        state_values = self.chatbot_graph.get_state(config).values
        conversation_name = state_values.get("conversation_name") # Safely get the name
        # print(f"state_values {state_values}") # Commented out debug print
        # Return the retrieved name or a default value
        return conversation_name if conversation_name else "New Conversation"

