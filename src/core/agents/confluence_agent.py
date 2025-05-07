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
from atlassian import Confluence
from langchain.retrievers import ConfluenceRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.schema import Document
import psycopg2
from psycopg2.extras import Json

@dataclass
class ConfluenceState(AgentState):
    """Extended state for Confluence agent."""
    page_cache: Dict[str, Any] = field(default_factory=dict)
    last_cache_update: float = 0.0
    current_search: Optional[str] = None
    search_results: Optional[List[Dict[str, Any]]] = None
    embedding_batch_size: int = 100
    retriever: Optional[ConfluenceRetriever] = None
    vector_store: Optional[PGVector] = None

class ConfluenceAgent(BaseAgent):
    def __init__(self, config: Config, agent_config: AgentConfig):
        super().__init__(config, agent_config)
        self.connection_params = agent_config.connection
        self.embedding_store = get_embedding_store(config.embedding)
        self.state = ConfluenceState(
            embedding_batch_size=agent_config.connection.get('embedding_batch_size', 100)
        )
        self.confluence = Confluence(
            url=self.connection_params['url'],
            username=self.connection_params['username'],
            password=self.connection_params['password']
        )
        self._initialize_retrievers()
        self.group_chat = self._create_group_chat()

    def _initialize_retrievers(self):
        """Initialize LangChain retrievers with pgvector."""
        try:
            # Initialize Confluence retriever
            self.state.retriever = ConfluenceRetriever(
                confluence_url=self.connection_params['url'],
                username=self.connection_params['username'],
                api_key=self.connection_params['password'],
                space_key=self.connection_params.get('space_key', ''),
                page_ids=self.connection_params.get('page_ids', []),
                include_attachments=False,
                include_comments=False
            )

            # Initialize text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            # Initialize embeddings
            embeddings = OpenAIEmbeddings(
                openai_api_key=self.config.llm.api_key
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
                collection_name="confluence_embeddings",
                pre_delete_collection=False
            )

        except Exception as e:
            logger.error(f"Error initializing retrievers: {str(e)}")
            raise

    def _create_group_chat(self) -> GroupChatManager:
        """Create a group chat with specialized agents."""
        # Create content expert agent
        content_expert = AssistantAgent(
            name="ContentExpert",
            llm_config=self.llm_config,
            system_message="""You are an expert in analyzing and understanding Confluence content.
            Your role is to:
            1. Analyze page content and structure
            2. Extract key information
            3. Identify relevant sections
            4. Summarize content effectively
            
            Use the provided tools to:
            1. Search for content
            2. Retrieve page details
            3. Analyze content structure
            4. Extract key information"""
        )

        # Create search expert agent
        search_expert = AssistantAgent(
            name="SearchExpert",
            llm_config=self.llm_config,
            system_message="""You are an expert in searching and retrieving Confluence content.
            Your role is to:
            1. Formulate effective search queries
            2. Filter and rank results
            3. Identify relevant content
            4. Suggest search improvements
            
            Use the provided tools to:
            1. Search content
            2. Filter results
            3. Rank relevance
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
                content_expert,
                search_expert,
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
        1. Searching and retrieving Confluence content
        2. Understanding and analyzing page content
        3. Extracting relevant information
        4. Providing clear and concise answers
        
        You have access to the following tools:
        1. search_content: Search for content in Confluence
        2. get_page: Get detailed page information
        3. get_space: Get space information
        4. analyze_content: Analyze and extract information from content
        
        When answering questions:
        1. First, search for relevant content
        2. Analyze and understand the content
        3. Extract key information
        4. Provide clear explanations
        5. Include relevant links when available
        6. Handle errors gracefully and suggest alternatives"""

    def _register_tools(self) -> Dict[str, Callable]:
        """Register tools for the agent."""
        return {
            "search_content": self.search_content,
            "get_page": self.get_page,
            "get_space": self.get_space,
            "analyze_content": self.analyze_content,
            "semantic_search": self.semantic_search,
            "get_relevant_documents": self.get_relevant_documents,
            "hybrid_search": self.hybrid_search
        }

    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        try:
            # Update state
            self.state.context = context or {}
            self.state.current_search = message
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

    async def search_content(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for content in Confluence."""
        try:
            # Get query embedding
            query_embedding = await self.embedding_store.get_embedding(query)
            
            # Search for similar content
            results = await self.embedding_store.search(
                query_embedding,
                top_k=limit,
                filter={"type": "confluence"}
            )
            
            # Update state
            self.state.search_results = results
            self.state.current_search = query
            
            return results
        except Exception as e:
            logger.error(f"Error searching content: {str(e)}")
            raise

    async def get_page(self, page_id: str) -> Dict[str, Any]:
        """Get detailed page information."""
        try:
            # Check cache first
            if page_id in self.state.page_cache:
                return self.state.page_cache[page_id]
            
            # Get page from Confluence
            page = self.confluence.get_page_by_id(
                page_id=page_id,
                expand='body.storage,version'
            )
            
            # Store in cache
            self.state.page_cache[page_id] = page
            
            return page
        except Exception as e:
            logger.error(f"Error getting page: {str(e)}")
            raise

    async def get_space(self, space_key: str) -> Dict[str, Any]:
        """Get space information."""
        try:
            return self.confluence.get_space(
                space_key=space_key,
                expand='description.plain,homepage'
            )
        except Exception as e:
            logger.error(f"Error getting space: {str(e)}")
            raise

    async def analyze_content(self, content: str) -> Dict[str, Any]:
        """Analyze and extract information from content."""
        try:
            # Get content embedding
            content_embedding = await self.embedding_store.get_embedding(content)
            
            # Store embedding
            await self.embedding_store.store_embeddings(
                [content_embedding],
                [{
                    "type": "confluence",
                    "content": content,
                    "embedding": content_embedding
                }]
            )
            
            return {
                "embedding": content_embedding,
                "length": len(content),
                "type": "confluence"
            }
        except Exception as e:
            logger.error(f"Error analyzing content: {str(e)}")
            raise

    async def semantic_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search using pgvector."""
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
            
            # Update state
            self.state.search_results = results
            self.state.current_search = query
            
            return results
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            raise

    async def get_relevant_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Get relevant documents using LangChain retriever."""
        try:
            # Get documents from retriever
            docs = self.state.retriever.get_relevant_documents(
                query=query,
                k=k
            )
            
            # Format results
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("source", "")
                })
            
            # Update state
            self.state.search_results = results
            self.state.current_search = query
            
            return results
        except Exception as e:
            logger.error(f"Error getting relevant documents: {str(e)}")
            raise

    async def hybrid_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and keyword search."""
        try:
            # Get semantic search results
            semantic_results = await self.semantic_search(query, k)
            
            # Get keyword search results
            keyword_results = await self.get_relevant_documents(query, k)
            
            # Combine and deduplicate results
            combined_results = {}
            for result in semantic_results + keyword_results:
                key = result.get("metadata", {}).get("page_id")
                if key and key not in combined_results:
                    combined_results[key] = result
            
            # Sort by score
            results = sorted(
                combined_results.values(),
                key=lambda x: x.get("score", 0.0),
                reverse=True
            )[:k]
            
            # Update state
            self.state.search_results = results
            self.state.current_search = query
            
            return results
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            raise

    async def _update_content_cache(self):
        """Update the content cache with embeddings using pgvector."""
        try:
            # Get all spaces
            spaces = self.confluence.get_all_spaces()
            
            for space in spaces['results']:
                # Get all pages in space
                pages = self.confluence.get_all_pages_from_space(
                    space=space['key'],
                    start=0,
                    limit=self.state.embedding_batch_size
                )
                
                for page in pages:
                    # Get page content
                    content = page.get('body', {}).get('storage', {}).get('value', '')
                    
                    # Store in cache
                    self.state.page_cache[page['id']] = page
                    
                    # Create document
                    doc = Document(
                        page_content=content,
                        metadata={
                            "page_id": page['id'],
                            "title": page['title'],
                            "space_key": space['key'],
                            "type": "confluence",
                            "url": f"{self.connection_params['url']}/pages/viewpage.action?pageId={page['id']}"
                        }
                    )
                    
                    # Add to vector store
                    self.state.vector_store.add_documents([doc])
            
            self.state.last_cache_update = time.time()
            
        except Exception as e:
            logger.error(f"Error updating content cache: {str(e)}")
            raise 