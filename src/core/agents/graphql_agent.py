from typing import Dict, Any, List, Optional
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport
from .base_agent import BaseAgent
from ..config import Config, AgentConfig
from loguru import logger
import json

class GraphQLAgent(BaseAgent):
    def __init__(self, config: Config, agent_config: AgentConfig):
        super().__init__(config, agent_config)
        self.endpoints = agent_config.connection.get("endpoints", {})
        self._schema_cache = {}
        self._last_schema_update = 0

    def _get_system_message(self) -> str:
        return f"""You are {self.agent_config.name}, {self.agent_config.description}.
        You are an expert at:
        1. Understanding GraphQL schemas and types
        2. Converting natural language questions into GraphQL queries
        3. Working with multiple GraphQL endpoints
        4. Analyzing and explaining query results
        
        When answering questions:
        1. First, understand the relevant GraphQL schema
        2. Generate appropriate GraphQL queries
        3. Execute queries and analyze results
        4. Provide clear explanations with the data
        5. Include the GraphQL query used when relevant
        6. Handle errors gracefully and suggest alternatives"""

    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        try:
            # Update schema cache if needed
            await self._update_schema_cache()
            
            # Get relevant schema information
            schema_info = self._get_relevant_schema(message)
            
            # Prepare context with schema information
            enhanced_context = {
                "schema_info": schema_info,
                **(context or {})
            }

            # Process the message with enhanced context
            return await super().process_message(message, enhanced_context)

        except Exception as e:
            return await self.handle_error(e)

    async def _update_schema_cache(self):
        """Update the schema cache if it's expired."""
        current_time = time.time()
        if current_time - self._last_schema_update > self.agent_config.schema_cache_ttl:
            try:
                for endpoint_name, endpoint_config in self.endpoints.items():
                    transport = RequestsHTTPTransport(
                        url=endpoint_config["url"],
                        headers=endpoint_config.get("headers", {}),
                        verify=endpoint_config.get("verify", True)
                    )
                    
                    client = Client(
                        transport=transport,
                        fetch_schema_from_transport=True
                    )
                    
                    # Get the schema
                    schema = client.schema
                    self._schema_cache[endpoint_name] = {
                        "types": self._extract_types(schema),
                        "queries": self._extract_queries(schema),
                        "mutations": self._extract_mutations(schema)
                    }
                
                self._last_schema_update = current_time
                
            except Exception as e:
                logger.error(f"Error updating schema cache: {str(e)}")
                raise

    def _extract_types(self, schema) -> Dict[str, Any]:
        """Extract type information from the schema."""
        types = {}
        for type_name, type_info in schema.type_map.items():
            if not type_name.startswith("__"):
                types[type_name] = {
                    "kind": type_info.kind,
                    "fields": self._extract_fields(type_info)
                }
        return types

    def _extract_fields(self, type_info) -> Dict[str, Any]:
        """Extract field information from a type."""
        fields = {}
        if hasattr(type_info, "fields"):
            for field_name, field_info in type_info.fields.items():
                fields[field_name] = {
                    "type": str(field_info.type),
                    "args": self._extract_args(field_info)
                }
        return fields

    def _extract_args(self, field_info) -> Dict[str, Any]:
        """Extract argument information from a field."""
        args = {}
        if hasattr(field_info, "args"):
            for arg_name, arg_info in field_info.args.items():
                args[arg_name] = {
                    "type": str(arg_info.type),
                    "default_value": arg_info.default_value
                }
        return args

    def _extract_queries(self, schema) -> Dict[str, Any]:
        """Extract query information from the schema."""
        return self._extract_fields(schema.query_type) if schema.query_type else {}

    def _extract_mutations(self, schema) -> Dict[str, Any]:
        """Extract mutation information from the schema."""
        return self._extract_fields(schema.mutation_type) if schema.mutation_type else {}

    def _get_relevant_schema(self, query: str) -> Dict[str, Any]:
        """Get schema information relevant to the query."""
        # This is a simplified version - in practice, you'd want to use
        # semantic search to find relevant types and fields
        return self._schema_cache

    async def execute_query(self, endpoint_name: str, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a GraphQL query and return the results."""
        try:
            endpoint_config = self.endpoints[endpoint_name]
            transport = RequestsHTTPTransport(
                url=endpoint_config["url"],
                headers=endpoint_config.get("headers", {}),
                verify=endpoint_config.get("verify", True)
            )
            
            client = Client(
                transport=transport,
                fetch_schema_from_transport=True
            )
            
            # Execute the query
            result = await client.execute_async(
                gql(query),
                variable_values=variables
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing GraphQL query: {str(e)}")
            raise 