from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from databricks import sql
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from autogen.agentchat.contrib.text_analyzer_agent import TextAnalyzerAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from .base_agent import BaseAgent, AgentState
from ..config import Config, AgentConfig
from ..embeddings import get_embedding_store
from loguru import logger
import json
import time
import aiohttp
from graphql import build_schema, parse, validate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import psycopg2
from psycopg2.extras import Json

@dataclass
class DatabricksState(AgentState):
    """Extended state for Databricks agent."""
    schema_cache: Dict[str, Any] = field(default_factory=dict)
    last_schema_update: float = 0.0
    current_query: Optional[str] = None
    query_results: Optional[Dict[str, Any]] = None
    schema_cache_ttl: int = 3600  # 1 hour
    vector_store: Optional[PGVector] = None

class DatabricksAgent(BaseAgent):
    def __init__(self, config: Config, agent_config: AgentConfig):
        super().__init__(config, agent_config)
        self.connection_params = agent_config.connection
        self.embedding_store = get_embedding_store(config.embedding)
        self.state = DatabricksState(
            schema_cache_ttl=agent_config.connection.get('schema_cache_ttl', 3600)
        )
        self._initialize_vector_store()
        self.group_chat = self._create_group_chat()

    def _initialize_vector_store(self):
        """Initialize pgvector store for schema embeddings."""
        try:
            # Initialize embeddings
            embeddings = OpenAIEmbeddings(
                openai_api_key=self.config.llm.api_key
            )

            # Initialize text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            # Initialize pgvector store
            connection_string = PGVector.connection_string_from_db_params(
                driver="psycopg2",
                host=self.config.embedding.connection.host,
                port=self.config.embedding.connection.port,
                database=self.config.embedding.connection.database,
                user=self.config.embedding.connection.user,
                password=self.config.embedding.connection.password
            )

            self.state.vector_store = PGVector(
                connection_string=connection_string,
                embedding_function=embeddings,
                collection_name="databricks_schema_embeddings",
                pre_delete_collection=False
            )

        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def _create_group_chat(self) -> GroupChatManager:
        """Create a group chat with specialized agents."""
        # Create SQL expert agent
        sql_expert = AssistantAgent(
            name="SQLExpert",
            llm_config=self.llm_config,
            system_message="""You are an expert in SQL and database schema design.
            Your role is to:
            1. Analyze database schemas
            2. Generate efficient SQL queries
            3. Validate query correctness
            4. Suggest query optimizations
            
            Use the provided tools to:
            1. Get schema information
            2. Execute queries
            3. Validate results
            4. Optimize performance
            
            Always consider performance and best practices."""
        )

        # Create schema expert agent
        schema_expert = AssistantAgent(
            name="SchemaExpert",
            llm_config=self.llm_config,
            system_message="""You are an expert in database schema analysis.
            Your role is to:
            1. Understand table relationships
            2. Identify relevant tables and columns
            3. Explain schema structure
            4. Suggest appropriate joins and filters
            
            Use the provided tools to:
            1. Search schema information
            2. Analyze table relationships
            3. Validate schema access
            4. Explain schema structure"""
        )

        # Create query validator agent
        query_validator = AssistantAgent(
            name="QueryValidator",
            llm_config=self.llm_config,
            system_message="""You are an expert in SQL query validation.
            Your role is to:
            1. Check query syntax
            2. Verify table and column existence
            3. Validate query logic
            4. Suggest improvements
            
            Use the provided tools to:
            1. Validate query syntax
            2. Check table and column existence
            3. Analyze query performance
            4. Suggest optimizations"""
        )

        # Create text analyzer agent
        text_analyzer = TextAnalyzerAgent(
            name="TextAnalyzer",
            llm_config=self.llm_config,
            system_message="""You are an expert at analyzing text and determining its intent.
            Your role is to:
            1. Analyze user questions
            2. Identify key topics and requirements
            3. Determine which agents are needed
            4. Extract relevant context
            5. Guide the conversation flow"""
        )

        # Create group chat
        group_chat = GroupChat(
            agents=[
                self.agent,
                sql_expert,
                schema_expert,
                query_validator,
                text_analyzer
            ],
            messages=[],
            max_round=self.agent_config.group_chat.max_round,
            speaker_selection_method="round_robin"
        )

        return GroupChatManager(
            groupchat=group_chat,
            llm_config=self.llm_config
        )

    def _get_system_message(self) -> str:
        return f"""You are {self.agent_config.name}, {self.agent_config.description}.
        You are an expert at:
        1. Understanding and querying Databricks SQL tables
        2. Working with Databricks system tables
        3. Converting natural language questions into SQL queries
        4. Analyzing and explaining query results
        
        You have access to the following tools:
        1. get_schema_info: Get information about table schemas
        2. execute_query: Execute SQL queries
        3. get_system_table_info: Get information about system tables
        4. search_schema: Search for relevant tables and columns
        
        When answering questions:
        1. First, understand the schema of relevant tables
        2. Generate appropriate SQL queries
        3. Execute queries and analyze results
        4. Provide clear explanations with the data
        5. Include the SQL query used when relevant
        6. Handle errors gracefully and suggest alternatives"""

    def _register_tools(self) -> Dict[str, Callable]:
        """Register tools for the agent."""
        return {
            "get_schema": self.get_schema,
            "execute_query": self.execute_query,
            "validate_query": self.validate_query,
            "search_schema": self.search_schema,
            "semantic_schema_search": self.semantic_schema_search
        }

    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        try:
            # Update schema cache if needed
            await self._update_schema_cache()
            
            # Get relevant schema information
            schema_info = await self._get_relevant_schema(message)
            
            # Update state
            self.state.context = context or {}
            self.state.current_query = message
            self.state.messages.append({"role": "user", "content": message})

            # Prepare context with schema information
            enhanced_context = {
                "schema_info": schema_info,
                **(context or {})
            }

            # Create a user proxy agent for this interaction
            user_proxy = RetrieveUserProxyAgent(
                name="user_proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=0,
                code_execution_config={
                    "work_dir": "workspace",
                    "use_docker": False,
                    "last_n_messages": 3
                }
            )

            # Register tools
            for tool_name, tool_func in self.tools.items():
                user_proxy.register_function(
                    function_map={tool_name: tool_func}
                )

            # Initiate the group chat
            chat_result = await user_proxy.initiate_chat(
                self.group_chat,
                message=message,
                context=enhanced_context
            )

            # Update state with response
            response = chat_result.last_message()["content"]
            self.state.messages.append({"role": "assistant", "content": response})

            return response

        except Exception as e:
            return await self.handle_error(e)

    async def _get_schema_info(self, table_name: str) -> Dict[str, Any]:
        """Get schema information for a table."""
        try:
            if table_name not in self.state.schema_cache:
                await self._update_schema_cache()
            return self.state.schema_cache.get(table_name, {})
        except Exception as e:
            logger.error(f"Error getting schema info: {str(e)}")
            raise

    async def _search_schema(self, query: str) -> List[Dict[str, Any]]:
        """Search for relevant tables and columns using embeddings."""
        try:
            # Get query embedding
            query_embedding = await self.embedding_store.get_embedding(query)
            
            # Search for similar schema entries
            results = await self.embedding_store.search(
                query_embedding,
                top_k=5,
                filter={"type": "schema"}
            )
            
            return results
        except Exception as e:
            logger.error(f"Error searching schema: {str(e)}")
            raise

    async def _update_schema_cache(self):
        """Update the schema cache with embeddings using pgvector."""
        try:
            # Get schema from endpoint
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.connection_params['endpoint'],
                    json={'query': '{ __schema { types { name fields { name type { name kind ofType { name kind } } } } } }'}
                ) as response:
                    schema_data = await response.json()
            
            # Build schema
            schema = build_schema(schema_data['data']['__schema'])
            
            # Store in cache
            self.state.schema_cache = schema
            
            # Create documents for each type and field
            docs = []
            for type_name, type_info in schema.type_map.items():
                if not type_name.startswith("__"):
                    # Create document for type
                    type_doc = Document(
                        page_content=f"Type: {type_name}\nKind: {type_info.kind}",
                        metadata={
                            "type": "schema_type",
                            "name": type_name,
                            "kind": type_info.kind
                        }
                    )
                    docs.append(type_doc)
                    
                    # Create documents for fields
                    if hasattr(type_info, "fields"):
                        for field_name, field_info in type_info.fields.items():
                            field_doc = Document(
                                page_content=f"Field: {field_name}\nType: {field_info.type}\nParent Type: {type_name}",
                                metadata={
                                    "type": "schema_field",
                                    "name": field_name,
                                    "parent_type": type_name,
                                    "field_type": str(field_info.type)
                                }
                            )
                            docs.append(field_doc)
            
            # Add documents to vector store
            self.state.vector_store.add_documents(docs)
            
            self.state.last_schema_update = time.time()
            
        except Exception as e:
            logger.error(f"Error updating schema cache: {str(e)}")
            raise

    async def execute_query(self, query: str) -> Dict[str, Any]:
        """Execute a SQL query and return the results."""
        try:
            with sql.connect(**self.connection_params) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    results = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    
                    # Update state with query results
                    self.state.query_results = {
                        "columns": columns,
                        "rows": results,
                        "row_count": len(results)
                    }
                    
                    return self.state.query_results
                    
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise

    async def get_system_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a system table."""
        try:
            query = f"""
            SELECT *
            FROM system.{table_name}
            LIMIT 100
            """
            return await self.execute_query(query)
            
        except Exception as e:
            logger.error(f"Error getting system table info: {str(e)}")
            raise

    async def semantic_schema_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search on schema using pgvector."""
        try:
            # Search vector store
            docs = self.state.vector_store.similarity_search(
                query=query,
                k=k
            )
            
            # Format results
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": doc.metadata.get("score", 0.0)
                })
            
            return results
        except Exception as e:
            logger.error(f"Error in semantic schema search: {str(e)}")
            raise 