from typing import Dict, Any, Optional, List, Callable
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from autogen.agentchat.contrib.text_analyzer_agent import TextAnalyzerAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
import os
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from ..config import Config, AgentConfig

class BaseAgent:
    def __init__(self, config: Config, agent_config: AgentConfig):
        self.config = config
        self.agent_config = agent_config
        self.llm_config = self._create_llm_config()
        self.agent = self._create_agent()
        self.tools = self._register_tools()

    def _create_llm_config(self) -> Dict[str, Any]:
        """Create LLM configuration based on provider."""
        if self.config.llm.provider == "bedrock":
            return {
                "config_list": [{
                    "model": self.config.llm.model,
                    "api_key": "dummy",  # Will be handled by AWS credentials
                    "base_url": f"https://bedrock.{self.config.llm.region}.amazonaws.com",
                    "api_type": "bedrock",
                    "api_version": "2023-09-30"
                }],
                "temperature": self.config.llm.temperature,
                "max_tokens": self.config.llm.max_tokens,
                "cache_seed": None  # Disable caching for dynamic responses
            }
        elif self.config.llm.provider == "gemini":
            return {
                "config_list": [{
                    "model": self.config.llm.model,
                    "api_key": os.getenv("GOOGLE_API_KEY"),
                    "base_url": "https://generativelanguage.googleapis.com",
                    "api_type": "gemini"
                }],
                "temperature": self.config.llm.temperature,
                "max_tokens": self.config.llm.max_tokens,
                "cache_seed": None
            }
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm.provider}")

    def _create_agent(self) -> AssistantAgent:
        """Create the AutoGen agent instance."""
        return AssistantAgent(
            name=self.agent_config.name,
            llm_config=self.llm_config,
            system_message=self._get_system_message(),
            human_input_mode="NEVER",
            max_consecutive_auto_reply=self.agent_config.max_iterations,
            code_execution_config={
                "work_dir": "workspace",
                "use_docker": False,
                "last_n_messages": 3
            }
        )

    def _get_system_message(self) -> str:
        """Get the system message for the agent."""
        return f"""You are {self.agent_config.name}, {self.agent_config.description}.
        You are part of a multi-agent system that helps users retrieve information and answer questions.
        Always be helpful, accurate, and concise in your responses.
        Use the provided tools to accomplish tasks.
        When using tools, explain your reasoning and show your work."""

    def _register_tools(self) -> Dict[str, Callable]:
        """Register tools for the agent."""
        return {}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Process a message and return a response."""
        try:
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

            # Add context to the message if provided
            if context:
                message = f"Context: {context}\n\nUser message: {message}"

            # Initiate the chat
            chat_result = await user_proxy.initiate_chat(
                self.agent,
                message=message
            )

            return chat_result.last_message()["content"]

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            raise

    async def handle_error(self, error: Exception) -> str:
        """Handle errors gracefully."""
        error_message = f"I apologize, but I encountered an error: {str(error)}. Please try again or rephrase your question."
        logger.error(f"Agent error: {str(error)}")
        return error_message

    def create_group_chat(self, agents: List[AssistantAgent], max_round: int = 10) -> GroupChatManager:
        """Create a group chat with the specified agents."""
        group_chat = GroupChat(
            agents=agents,
            messages=[],
            max_round=max_round,
            speaker_selection_method="round_robin"
        )
        return GroupChatManager(
            groupchat=group_chat,
            llm_config=self.llm_config
        ) 