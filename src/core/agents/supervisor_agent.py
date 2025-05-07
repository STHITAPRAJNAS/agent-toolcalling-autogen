from typing import Dict, Any, List, Optional
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from autogen.agentchat.contrib.text_analyzer_agent import TextAnalyzerAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from .base_agent import BaseAgent
from .confluence_agent import ConfluenceAgent
from .databricks_agent import DatabricksAgent
from .graphql_agent import GraphQLAgent
from ..config import Config, AgentConfig
from ..memory import ConversationMemory
from loguru import logger

class SupervisorAgent(BaseAgent):
    def __init__(self, config: Config, agent_config: AgentConfig):
        super().__init__(config, agent_config)
        self.confluence_agent = ConfluenceAgent(config, config.agents["confluence"])
        self.databricks_agent = DatabricksAgent(config, config.agents["databricks"])
        self.graphql_agent = GraphQLAgent(config, config.agents["graphql"])
        self.memory = ConversationMemory(config.memory)
        self.group_chat = self._create_group_chat()

    def _create_group_chat(self) -> GroupChatManager:
        """Create a group chat with all specialized agents."""
        # Create coordinator agent
        coordinator = AssistantAgent(
            name="Coordinator",
            llm_config=self.llm_config,
            system_message="""You are the coordinator of a team of specialized agents.
            Your role is to:
            1. Analyze user questions to determine which agent(s) should handle them
            2. Coordinate between agents when multiple are needed
            3. Synthesize responses from multiple agents
            4. Maintain conversation context
            5. Handle errors and retry with different agents if needed
            
            Use the provided tools to:
            1. Search conversation history
            2. Get relevant context
            3. Coordinate between agents
            4. Synthesize responses"""
        )

        # Create synthesizer agent
        synthesizer = AssistantAgent(
            name="Synthesizer",
            llm_config=self.llm_config,
            system_message="""You are an expert at synthesizing information from multiple sources.
            Your role is to:
            1. Combine responses from different agents
            2. Resolve conflicts in information
            3. Create coherent and comprehensive answers
            4. Maintain consistency in responses
            5. Format responses appropriately
            
            Use the provided tools to:
            1. Access conversation history
            2. Search for relevant context
            3. Format and structure responses"""
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
                coordinator,
                synthesizer,
                text_analyzer,
                self.confluence_agent.agent,
                self.databricks_agent.agent,
                self.graphql_agent.agent
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
        You are the supervisor of a team of specialized agents:
        1. ConfluenceExpert: Expert in retrieving information from Confluence
        2. DatabricksExpert: Expert in Databricks SQL and system tables
        3. GraphQLExpert: Expert in GraphQL queries and schema
        
        Your responsibilities:
        1. Analyze user questions to determine which agent(s) should handle them
        2. Coordinate between agents when multiple are needed
        3. Synthesize responses from multiple agents when necessary
        4. Maintain conversation context and history
        5. Handle errors and retry with different agents if needed
        
        You have access to the following tools:
        1. get_conversation_history: Get conversation history
        2. add_to_history: Add message to conversation history
        3. clear_history: Clear conversation history
        4. search_history: Search through conversation history
        
        When processing questions:
        1. First, determine the type of information needed
        2. Select the most appropriate agent(s)
        3. Provide necessary context from conversation history
        4. Combine and format responses coherently
        5. Handle follow-up questions appropriately"""

    def _register_tools(self) -> Dict[str, Callable]:
        """Register tools for the agent."""
        return {
            "get_conversation_history": self.memory.get_conversation_history,
            "add_to_history": self.memory.add_to_history,
            "clear_history": self.memory.clear_history,
            "search_history": self.memory.search_history
        }

    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        try:
            # Get conversation history
            conversation_id = context.get("conversation_id") if context else None
            user_id = context.get("user_id") if context else None
            
            if conversation_id and user_id:
                history = await self.memory.get_conversation_history(conversation_id)
            else:
                history = []

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

            # Prepare context with conversation history
            enhanced_context = {
                "conversation_history": history,
                **(context or {})
            }

            # Initiate the group chat
            chat_result = await user_proxy.initiate_chat(
                self.group_chat,
                message=message,
                context=enhanced_context
            )

            # Get the final response
            final_response = chat_result.last_message()["content"]

            # Store in conversation history
            if conversation_id and user_id:
                await self.memory.add_to_history(
                    conversation_id,
                    user_id,
                    {
                        "role": "user",
                        "content": message
                    }
                )
                await self.memory.add_to_history(
                    conversation_id,
                    user_id,
                    {
                        "role": "assistant",
                        "content": final_response
                    }
                )

            return final_response

        except Exception as e:
            return await self.handle_error(e)

    async def handle_error(self, error: Exception) -> str:
        """Handle errors gracefully."""
        error_message = f"I apologize, but I encountered an error: {str(error)}. Please try again or rephrase your question."
        logger.error(f"Supervisor error: {str(error)}")
        return error_message 