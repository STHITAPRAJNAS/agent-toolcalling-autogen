from typing import Annotated, Dict, List, TypedDict, Optional, Any, Union, Callable, Tuple
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    FunctionMessage,
    BaseMessage,
    ToolMessage
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_postgres import PostgresSaver
from langchain_core.runnables import RunnablePassthrough
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool
from langchain_core.cache import InMemoryCache
from pydantic import BaseModel, Field
from loguru import logger
import json
from dataclasses import dataclass
from jaydebeapi import connect
from enum import Enum
from langflow import CustomComponent
from langflow.interface.custom.custom_component import CustomComponent
from langflow.interface.types.base import BaseType
from langflow.interface.types.agent import AgentType
import time
from functools import wraps
import hashlib

class IntentType(str, Enum):
    """Types of user intents."""
    CONFLUENCE = "confluence"
    DATABRICKS = "databricks"
    BOTH = "both"
    UNKNOWN = "unknown"

class ActionType(str, Enum):
    """Types of agent actions."""
    RETRIEVE = "retrieve"
    GENERATE_SQL = "generate_sql"
    EXECUTE_SQL = "execute_sql"
    GENERATE_RESPONSE = "generate_response"
    ERROR_HANDLING = "error_handling"

class ToolCallPattern(str, Enum):
    """Patterns for tool calling."""
    SEQUENTIAL = "sequential"  # Call tools one after another
    PARALLEL = "parallel"      # Call multiple tools simultaneously
    CONDITIONAL = "conditional"  # Call tools based on conditions
    RETRY = "retry"           # Retry failed tool calls
    CACHED = "cached"         # Use cached results when available

@dataclass
class DatabricksConfig:
    """Configuration for Databricks connection."""
    host: str
    port: int
    http_path: str
    access_token: str
    catalog: str
    schema: str

class AgentState(TypedDict):
    """State for the agent workflow."""
    messages: List[BaseMessage]
    current_question: str
    rephrased_question: Optional[str]
    intent: Optional[IntentType]
    confluence_context: List[Document]
    databricks_context: List[Document]
    sql_query: Optional[str]
    sql_result: Optional[Any]
    final_answer: Optional[str]
    error: Optional[str]
    attempt_count: int
    tool_calls: List[Dict[str, Any]]
    cache_hits: int
    retry_count: Dict[str, int]
    needs_clarification: bool
    clarification_response: Optional[str]

class SQLQuery(BaseModel):
    """Model for SQL query generation."""
    query: str = Field(description="The SQL query to execute")
    explanation: str = Field(description="Explanation of what the query does")

class IntentClassification(BaseModel):
    """Model for intent classification."""
    intent: IntentType = Field(description="The classified intent of the question")
    confidence: float = Field(description="Confidence score of the classification")
    explanation: str = Field(description="Explanation of the classification")

class ToolCallResult(BaseModel):
    """Model for tool call results."""
    success: bool
    result: Any
    error: Optional[str]
    cache_hit: bool
    retry_count: int
    execution_time: float

def cache_tool_result(ttl: int = 3600):
    """Decorator to cache tool results."""
    cache = InMemoryCache()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = hashlib.md5(
                json.dumps({
                    "func": func.__name__,
                    "args": args,
                    "kwargs": kwargs
                }).encode()
            ).hexdigest()
            
            # Check cache
            cached_result = cache.lookup(key)
            if cached_result:
                return ToolCallResult(
                    success=True,
                    result=cached_result,
                    cache_hit=True,
                    retry_count=0,
                    execution_time=0
                )
            
            # Execute tool
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Cache result
                cache.update(key, result, ttl)
                
                return ToolCallResult(
                    success=True,
                    result=result,
                    cache_hit=False,
                    retry_count=0,
                    execution_time=execution_time
                )
            except Exception as e:
                execution_time = time.time() - start_time
                return ToolCallResult(
                    success=False,
                    result=None,
                    error=str(e),
                    cache_hit=False,
                    retry_count=0,
                    execution_time=execution_time
                )
        return wrapper
    return decorator

def retry_tool(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry failed tool calls."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            while retry_count < max_retries:
                try:
                    result = func(*args, **kwargs)
                    return ToolCallResult(
                        success=True,
                        result=result,
                        cache_hit=False,
                        retry_count=retry_count,
                        execution_time=0
                    )
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        return ToolCallResult(
                            success=False,
                            result=None,
                            error=str(e),
                            cache_hit=False,
                            retry_count=retry_count,
                            execution_time=0
                        )
                    time.sleep(delay * retry_count)
            return None
        return wrapper
    return decorator

class LangGraphAgent(CustomComponent):
    """A LangFlow-compatible agent that uses LangGraph for dynamic workflows."""
    
    display_name: str = "LangGraph Agent"
    description: str = "An agentic workflow using LangGraph with tool calling"
    
    # Example prompts for different use cases
    EXAMPLE_PROMPTS = {
        "confluence_search": """Search Confluence for information about {topic}.
Focus on the most recent and relevant documents.""",
        
        "databricks_analysis": """Analyze the data in Databricks to answer: {question}
Use the system tables to understand the schema and generate appropriate SQL.""",
        
        "combined_query": """Find information about {topic} in Confluence and analyze related data in Databricks.
First search documentation, then use that context to query the data."""
    }
    
    def __init__(
        self,
        llm,
        confluence_retriever: BaseRetriever,
        databricks_retriever: BaseRetriever,
        databricks_config: DatabricksConfig,
        postgres_connection_string: str,
        max_attempts: int = 3,
        cache_ttl: int = 3600,
        retry_delay: float = 1.0
    ):
        super().__init__()
        self.llm = llm
        self.confluence_retriever = confluence_retriever
        self.databricks_retriever = databricks_retriever
        self.databricks_config = databricks_config
        self.max_attempts = max_attempts
        self.cache_ttl = cache_ttl
        self.retry_delay = retry_delay
        
        # Initialize PostgreSQL checkpointer
        self.checkpointer = PostgresSaver(
            connection_string=postgres_connection_string,
            table_name="agent_state"
        )
        
        # Initialize tools
        self.tools = self._create_tools()
        
        # Initialize the graph
        self.graph = self._build_graph()
    
    def _create_tools(self) -> List[BaseTool]:
        """Create tools for the agent."""
        
        @tool
        @cache_tool_result(ttl=self.cache_ttl)
        @retry_tool(max_retries=self.max_attempts, delay=self.retry_delay)
        def search_confluence(query: str) -> str:
            """Search Confluence for relevant information."""
            try:
                docs = self.confluence_retriever.get_relevant_documents(query)
                return json.dumps([doc.page_content for doc in docs], indent=2)
            except Exception as e:
                return f"Error searching Confluence: {str(e)}"
        
        @tool
        @cache_tool_result(ttl=self.cache_ttl)
        @retry_tool(max_retries=self.max_attempts, delay=self.retry_delay)
        def search_databricks(query: str) -> str:
            """Search Databricks system tables for relevant information."""
            try:
                docs = self.databricks_retriever.get_relevant_documents(query)
                return json.dumps([doc.page_content for doc in docs], indent=2)
            except Exception as e:
                return f"Error searching Databricks: {str(e)}"
        
        @tool
        @cache_tool_result(ttl=self.cache_ttl)
        @retry_tool(max_retries=self.max_attempts, delay=self.retry_delay)
        def generate_sql(query: str, context: str) -> str:
            """Generate a SQL query based on the question and context."""
            try:
                system_message = SystemMessage(content=f"""You are a SQL expert. Generate a SQL query based on the following context and question.
Context: {context}
Question: {query}

Generate a SQL query that answers the question. Include an explanation of what the query does.""")
                
                messages = [
                    system_message,
                    HumanMessage(content=query)
                ]
                
                response = self.llm.invoke(messages)
                sql_query = SQLQuery.parse_raw(response.content)
                return json.dumps({
                    "query": sql_query.query,
                    "explanation": sql_query.explanation
                }, indent=2)
            except Exception as e:
                return f"Error generating SQL: {str(e)}"
        
        @tool
        @cache_tool_result(ttl=self.cache_ttl)
        @retry_tool(max_retries=self.max_attempts, delay=self.retry_delay)
        def execute_sql(query: str) -> str:
            """Execute a SQL query against Databricks."""
            try:
                conn = connect(
                    "com.databricks.client.jdbc.Driver",
                    f"jdbc:databricks://{self.databricks_config.host}:{self.databricks_config.port};"
                    f"HttpPath={self.databricks_config.http_path};"
                    f"Catalog={self.databricks_config.catalog};"
                    f"Schema={self.databricks_config.schema}",
                    [self.databricks_config.access_token],
                    "path/to/databricks-jdbc-driver.jar"
                )
                
                cursor = conn.cursor()
                cursor.execute(query)
                results = cursor.fetchall()
                
                columns = [desc[0] for desc in cursor.description]
                formatted_results = [dict(zip(columns, row)) for row in results]
                
                return json.dumps(formatted_results, indent=2)
            except Exception as e:
                return f"Error executing SQL: {str(e)}"
            finally:
                if 'conn' in locals():
                    conn.close()
        
        return [
            search_confluence,
            search_databricks,
            generate_sql,
            execute_sql
        ]
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with tool calling."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("process_input", self._process_input)
        workflow.add_node("rephrase_question", self._rephrase_question)
        workflow.add_node("classify_intent", self._classify_intent)
        workflow.add_node("agent_step", self._agent_step)
        workflow.add_node("handle_tool_response", self._handle_tool_response)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("clarify_question", self._clarify_question)
        
        # Add edges
        workflow.add_edge(START, "process_input")
        workflow.add_edge("process_input", "rephrase_question")
        workflow.add_edge("rephrase_question", "classify_intent")
        
        # Add conditional edges based on intent
        workflow.add_conditional_edges(
            "classify_intent",
            self._route_by_intent,
            {
                "confluence": "agent_step",
                "databricks": "agent_step",
                "both": "agent_step",
                "clarify": "clarify_question"
            }
        )
        
        # Add edges for tool execution
        workflow.add_edge("agent_step", "handle_tool_response")
        workflow.add_edge("handle_tool_response", "agent_step")
        workflow.add_edge("agent_step", "generate_response")
        workflow.add_edge("clarify_question", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _process_input(self, state: AgentState) -> AgentState:
        """Process the user input."""
        logger.info("Processing input")
        
        # Get the last message
        last_message = state["messages"][-1]
        state["current_question"] = last_message.content
        
        # Initialize state
        state["rephrased_question"] = None
        state["intent"] = None
        state["confluence_context"] = []
        state["databricks_context"] = []
        state["sql_query"] = None
        state["sql_result"] = None
        state["final_answer"] = None
        state["error"] = None
        state["attempt_count"] = 0
        state["tool_calls"] = []
        state["cache_hits"] = 0
        state["retry_count"] = {}
        state["needs_clarification"] = False
        state["clarification_response"] = None
        
        return state
    
    def _rephrase_question(self, state: AgentState) -> AgentState:
        """Rephrase the question for better retrieval."""
        logger.info("Rephrasing question")
        
        system_prompt = """You are a helpful assistant that rephrases the user's question to be a standalone question optimized for retrieval.
Do not try to expand unknown acronyms. If there are pronouns in the question please use the conversation history to get context.
Make sure you return the original question if it cannot be rephrased."""
        
        messages = [
            SystemMessage(content=system_prompt),
            state["messages"][-1]
        ]
        
        prompt = ChatPromptTemplate.from_messages(messages)
        rephrased = self.llm.invoke(prompt)
        state["rephrased_question"] = rephrased.content.strip()
        
        return state
    
    def _classify_intent(self, state: AgentState) -> AgentState:
        """Classify the intent of the question."""
        logger.info("Classifying intent")
        
        system_prompt = """You are an expert at classifying questions. Determine if the question is about:
1. Confluence documentation
2. Databricks data
3. Both
4. Needs clarification

Return your classification with confidence score and explanation."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["rephrased_question"])
        ]
        
        response = self.llm.invoke(messages)
        classification = IntentClassification.parse_raw(response.content)
        state["intent"] = classification.intent
        
        return state
    
    def _route_by_intent(self, state: AgentState) -> str:
        """Route the workflow based on intent."""
        intent = state["intent"]
        if intent == IntentType.CONFLUENCE:
            return "confluence"
        elif intent == IntentType.DATABRICKS:
            return "databricks"
        elif intent == IntentType.BOTH:
            return "both"
        else:
            return "clarify"
    
    def _agent_step(self, state: AgentState) -> AgentState:
        """Execute one step of the agent's reasoning."""
        logger.info("Executing agent step")
        
        # Create system message with example prompts
        system_message = SystemMessage(content="""You are an expert at answering questions using available tools.
You have access to the following tools:
1. search_confluence: Search Confluence for relevant information
2. search_databricks: Search Databricks system tables for relevant information
3. generate_sql: Generate SQL queries based on context
4. execute_sql: Execute SQL queries against Databricks

Example patterns for different types of questions:
1. For documentation questions:
   - Use search_confluence to find relevant documentation
   - Focus on the most recent and relevant documents

2. For data analysis questions:
   - Use search_databricks to understand the schema
   - Generate and execute SQL queries to analyze the data

3. For combined questions:
   - First search documentation for context
   - Then use that context to query the data
   - Combine both sources in your response

Use these tools to gather information and answer the question.
If you have enough information to provide a final answer, say so.""")
        
        # Create messages for the LLM
        messages = [
            system_message,
            HumanMessage(content=state["rephrased_question"])
        ]
        
        # Add tool results if available
        for msg in state["messages"]:
            if isinstance(msg, ToolMessage):
                messages.append(msg)
        
        # Get agent's response
        response = self.llm.invoke(messages)
        state["messages"].append(response)
        
        return state
    
    def _handle_tool_response(self, state: AgentState) -> AgentState:
        """Handle tool responses and update state."""
        logger.info("Handling tool response")
        
        # Get the last message
        last_message = state["messages"][-1]
        
        # Check if the message contains tool calls
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                # Find the tool
                tool = next((t for t in self.tools if t.name == tool_call.name), None)
                if tool:
                    try:
                        # Execute the tool
                        result = tool.invoke(tool_call.args)
                        
                        # Track tool call
                        state["tool_calls"].append({
                            "name": tool_call.name,
                            "args": tool_call.args,
                            "result": result,
                            "cache_hit": getattr(result, "cache_hit", False),
                            "retry_count": getattr(result, "retry_count", 0)
                        })
                        
                        # Update cache hits and retry counts
                        if getattr(result, "cache_hit", False):
                            state["cache_hits"] += 1
                        state["retry_count"][tool_call.name] = getattr(result, "retry_count", 0)
                        
                        # Create tool message
                        tool_message = ToolMessage(
                            content=result.result if hasattr(result, "result") else result,
                            tool_call_id=tool_call.id,
                            name=tool_call.name
                        )
                        state["messages"].append(tool_message)
                        
                        # Update state based on tool
                        if tool_call.name == "search_confluence":
                            state["confluence_context"] = json.loads(result.result if hasattr(result, "result") else result)
                        elif tool_call.name == "search_databricks":
                            state["databricks_context"] = json.loads(result.result if hasattr(result, "result") else result)
                        elif tool_call.name == "generate_sql":
                            sql_info = json.loads(result.result if hasattr(result, "result") else result)
                            state["sql_query"] = sql_info["query"]
                        elif tool_call.name == "execute_sql":
                            state["sql_result"] = json.loads(result.result if hasattr(result, "result") else result)
                            
                    except Exception as e:
                        logger.error(f"Error executing tool {tool_call.name}: {e}")
                        state["error"] = f"Error executing tool {tool_call.name}: {str(e)}"
        
        return state
    
    def _clarify_question(self, state: AgentState) -> AgentState:
        """Generate a clarification question."""
        logger.info("Generating clarification question")
        
        system_prompt = """You are a helpful assistant that asks clarifying questions when the user's question is unclear.
Generate a question that will help understand what the user is looking for.
Focus on the most important aspects that need clarification."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["rephrased_question"])
        ]
        
        response = self.llm.invoke(messages)
        state["clarification_response"] = response.content
        state["needs_clarification"] = True
        
        return state
    
    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate final response."""
        logger.info("Generating response")
        
        # Create system message
        system_message = SystemMessage(content="""You are a helpful assistant. 
Use the available information to provide a clear and concise answer to the user's question.
If there were any errors, explain them in your response.
Include relevant statistics about tool usage (cache hits, retries) if available.""")
        
        # Create messages for the LLM
        messages = [
            system_message,
            HumanMessage(content=state["rephrased_question"])
        ]
        
        # Add all messages from the conversation
        messages.extend(state["messages"])
        
        # Get final response
        response = self.llm.invoke(messages)
        state["final_answer"] = response.content
        
        return state
    
    def invoke(self, message: str, config: Optional[Dict] = None) -> Dict:
        """Invoke the agent with a message."""
        if config is None:
            config = {}
            
        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "current_question": message,
            "rephrased_question": None,
            "intent": None,
            "confluence_context": [],
            "databricks_context": [],
            "sql_query": None,
            "sql_result": None,
            "final_answer": None,
            "error": None,
            "attempt_count": 0,
            "tool_calls": [],
            "cache_hits": 0,
            "retry_count": {},
            "needs_clarification": False,
            "clarification_response": None
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state, config)
        
        return {
            "answer": result["final_answer"],
            "intent": result["intent"],
            "confluence_context": result["confluence_context"],
            "databricks_context": result["databricks_context"],
            "sql_query": result["sql_query"],
            "sql_result": result["sql_result"],
            "error": result["error"],
            "tool_calls": result["tool_calls"],
            "cache_hits": result["cache_hits"],
            "retry_count": result["retry_count"],
            "needs_clarification": result["needs_clarification"],
            "clarification_response": result["clarification_response"]
        }
    
    async def astream(self, message: str, config: Optional[Dict] = None):
        """Stream the agent's response."""
        if config is None:
            config = {}
            
        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "current_question": message,
            "rephrased_question": None,
            "intent": None,
            "confluence_context": [],
            "databricks_context": [],
            "sql_query": None,
            "sql_result": None,
            "final_answer": None,
            "error": None,
            "attempt_count": 0,
            "tool_calls": [],
            "cache_hits": 0,
            "retry_count": {},
            "needs_clarification": False,
            "clarification_response": None
        }
        
        # Stream the graph
        async for chunk in self.graph.astream(initial_state, config):
            yield chunk 