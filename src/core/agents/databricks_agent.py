from typing import Dict, Any, List, Optional
from databricks import sql
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from autogen.agentchat.contrib.text_analyzer_agent import TextAnalyzerAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from .base_agent import BaseAgent
from ..config import Config, AgentConfig
from ..embeddings import get_embedding_store
from loguru import logger
import json

class DatabricksAgent(BaseAgent):
    def __init__(self, config: Config, agent_config: AgentConfig):
        super().__init__(config, agent_config)
        self.connection_params = agent_config.connection
        self.embedding_store = get_embedding_store(config.embedding)
        self._schema_cache = {}
        self._last_schema_update = 0
        self.group_chat = self._create_group_chat()

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
            "get_schema_info": self._get_schema_info,
            "execute_query": self.execute_query,
            "get_system_table_info": self.get_system_table_info,
            "search_schema": self._search_schema
        }

    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        try:
            # Update schema cache if needed
            await self._update_schema_cache()
            
            # Get relevant schema information
            schema_info = await self._get_relevant_schema(message)
            
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

            return chat_result.last_message()["content"]

        except Exception as e:
            return await self.handle_error(e)

    async def _get_schema_info(self, table_name: str) -> Dict[str, Any]:
        """Get schema information for a table."""
        try:
            if table_name not in self._schema_cache:
                await self._update_schema_cache()
            return self._schema_cache.get(table_name, {})
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
        """Update the schema cache with embeddings."""
        try:
            with sql.connect(**self.connection_params) as conn:
                with conn.cursor() as cursor:
                    # Get all tables
                    cursor.execute("SHOW TABLES")
                    tables = cursor.fetchall()
                    
                    # Get schema for each table
                    for table in tables:
                        table_name = table[0]
                        cursor.execute(f"DESCRIBE {table_name}")
                        columns = cursor.fetchall()
                        
                        # Store schema info
                        schema_info = {
                            "table_name": table_name,
                            "columns": [
                                {
                                    "name": col[0],
                                    "type": col[1],
                                    "description": col[2] if len(col) > 2 else None
                                }
                                for col in columns
                            ]
                        }
                        self._schema_cache[table_name] = schema_info
                        
                        # Store schema embedding
                        schema_text = f"Table {table_name}: " + ", ".join(
                            f"{col['name']} ({col['type']})"
                            for col in schema_info["columns"]
                        )
                        await self.embedding_store.store_embeddings(
                            [await self.embedding_store.get_embedding(schema_text)],
                            [{
                                "type": "schema",
                                "table_name": table_name,
                                "content": schema_text,
                                "schema_info": schema_info
                            }]
                        )
            
            self._last_schema_update = time.time()
            
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
                    
                    return {
                        "columns": columns,
                        "rows": results,
                        "row_count": len(results)
                    }
                    
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