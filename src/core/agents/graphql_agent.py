from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
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

@dataclass
class GraphQLState(AgentState):
    """Extended state for GraphQL agent."""
    schema_cache: Dict[str, Any] = field(default_factory=dict)
    last_schema_update: float = 0.0
    current_query: Optional[str] = None
    query_results: Optional[Dict[str, Any]] = None
    schema_cache_ttl: int = 3600  # 1 hour

class GraphQLAgent(BaseAgent):
    def __init__(self, config: Config, agent_config: AgentConfig):
        super().__init__(config, agent_config)
        self.connection_params = agent_config.connection
        self.embedding_store = get_embedding_store(config.embedding)
        self.state = GraphQLState(
            schema_cache_ttl=agent_config.connection.get('schema_cache_ttl', 3600)
        )
        self.group_chat = self._create_group_chat()

    def _create_group_chat(self) -> GroupChatManager:
        """Create a group chat with specialized agents."""
        # Create schema expert agent
        schema_expert = AssistantAgent(
            name="SchemaExpert",
            llm_config=self.llm_config,
            system_message="""You are an expert in GraphQL schema analysis.
            Your role is to:
            1. Understand schema structure
            2. Identify relevant types and fields
            3. Explain schema relationships
            4. Suggest optimal queries
            
            Use the provided tools to:
            1. Get schema information
            2. Analyze type relationships
            3. Validate schema access
            4. Explain schema structure"""
        )

        # Create query expert agent
        query_expert = AssistantAgent(
            name="QueryExpert",
            llm_config=self.llm_config,
            system_message="""You are an expert in GraphQL query construction.
            Your role is to:
            1. Generate efficient queries
            2. Validate query syntax
            3. Optimize query performance
            4. Handle complex queries
            
            Use the provided tools to:
            1. Execute queries
            2. Validate syntax
            3. Analyze performance
            4. Suggest improvements"""
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
                schema_expert,
                query_expert,
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
        1. Understanding GraphQL schemas
        2. Converting natural language to GraphQL queries
        3. Executing and validating queries
        4. Analyzing query results
        
        You have access to the following tools:
        1. get_schema: Get GraphQL schema information
        2. execute_query: Execute GraphQL queries
        3. validate_query: Validate query syntax
        4. search_schema: Search for relevant types and fields
        
        When answering questions:
        1. First, understand the schema structure
        2. Generate appropriate queries
        3. Execute and validate queries
        4. Provide clear explanations
        5. Include the query used when relevant
        6. Handle errors gracefully and suggest alternatives"""

    def _register_tools(self) -> Dict[str, Callable]:
        """Register tools for the agent."""
        return {
            "get_schema": self.get_schema,
            "execute_query": self.execute_query,
            "validate_query": self.validate_query,
            "search_schema": self.search_schema
        }

    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        try:
            # Update schema cache if needed
            await self._update_schema_cache()
            
            # Update state
            self.state.context = context or {}
            self.state.current_query = message
            self.state.messages.append({"role": "user", "content": message})

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
                context=self.state.context
            )

            # Update state with response
            response = chat_result.last_message()["content"]
            self.state.messages.append({"role": "assistant", "content": response})

            return response

        except Exception as e:
            return await self.handle_error(e)

    async def get_schema(self) -> Dict[str, Any]:
        """Get GraphQL schema information."""
        try:
            # Check if schema cache is valid
            if (time.time() - self.state.last_schema_update) > self.state.schema_cache_ttl:
                await self._update_schema_cache()
            
            return self.state.schema_cache
        except Exception as e:
            logger.error(f"Error getting schema: {str(e)}")
            raise

    async def execute_query(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a GraphQL query."""
        try:
            # Validate query first
            await self.validate_query(query)
            
            # Execute query
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.connection_params['endpoint'],
                    json={
                        'query': query,
                        'variables': variables or {}
                    },
                    headers=self.connection_params.get('headers', {})
                ) as response:
                    result = await response.json()
                    
                    # Update state
                    self.state.query_results = result
                    self.state.current_query = query
                    
                    return result
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise

    async def validate_query(self, query: str) -> bool:
        """Validate GraphQL query syntax."""
        try:
            # Get schema
            schema = await self.get_schema()
            
            # Parse and validate query
            document = parse(query)
            validation_errors = validate(schema, document)
            
            if validation_errors:
                raise ValueError(f"Query validation errors: {validation_errors}")
            
            return True
        except Exception as e:
            logger.error(f"Error validating query: {str(e)}")
            raise

    async def search_schema(self, query: str) -> List[Dict[str, Any]]:
        """Search for relevant types and fields in the schema."""
        try:
            # Get query embedding
            query_embedding = await self.embedding_store.get_embedding(query)
            
            # Search for similar schema entries
            results = await self.embedding_store.search(
                query_embedding,
                top_k=5,
                filter={"type": "graphql_schema"}
            )
            
            return results
        except Exception as e:
            logger.error(f"Error searching schema: {str(e)}")
            raise

    async def _update_schema_cache(self):
        """Update the schema cache with embeddings."""
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
            
            # Store schema embedding
            schema_text = str(schema)
            await self.embedding_store.store_embeddings(
                [await self.embedding_store.get_embedding(schema_text)],
                [{
                    "type": "graphql_schema",
                    "content": schema_text,
                    "schema": schema
                }]
            )
            
            self.state.last_schema_update = time.time()
            
        except Exception as e:
            logger.error(f"Error updating schema cache: {str(e)}")
            raise 