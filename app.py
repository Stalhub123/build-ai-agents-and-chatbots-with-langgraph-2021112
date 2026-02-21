"""
Multi-Agent Chatbot Streamlit App
Combines Product QnA, Orders, and Routing agents in a single interface
"""

import streamlit as st
import pandas as pd
import uuid
import sys
import os
from typing import TypedDict, Annotated
import operator
import functools

# Setup Azure OpenAI
os.environ["AZURE_OPENAI_API_KEY"] = st.secrets.get("AZURE_OPENAI_API_KEY", "")
os.environ["AZURE_OPENAI_ENDPOINT"] = st.secrets.get("AZURE_OPENAI_ENDPOINT", "")

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

# =====================
# LOAD DATA AND SETUP
# =====================

@st.cache_resource
def load_data():
    """Load CSV data for products and orders"""
    product_pricing_df = pd.read_csv("data/Laptop pricing.csv")
    product_orders_df = pd.read_csv("data/Laptop Orders.csv")
    return product_pricing_df, product_orders_df

@st.cache_resource
def setup_models():
    """Initialize Azure OpenAI models"""
    try:
        model = AzureChatOpenAI(
            azure_deployment="gpt-4o",
            api_version="2023-03-15-preview",
            model="gpt-4o"
        )
        
        embedding = AzureOpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_version="2023-05-15"
        )
        return model, embedding
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        return None, None

@st.cache_resource
def setup_product_features_retriever():
    """Setup product features retrieval tool"""
    try:
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        
        from langchain.tools.retriever import create_retriever_tool
        from langchain_chroma import Chroma
        from langchain_community.document_loaders import PyPDFLoader
        
        _, embedding = setup_models()
        
        # Load PDF and create vector store
        loader = PyPDFLoader("./data/Laptop product descriptions.pdf")
        docs = loader.load()
        
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
        splits = text_splitter.split_documents(docs)
        
        prod_feature_store = Chroma.from_documents(
            documents=splits,
            embedding=embedding
        )
        
        get_product_features = create_retriever_tool(
            prod_feature_store.as_retriever(search_kwargs={"k": 1}),
            name="Get_Product_Features",
            description="""
            This store contains details about Laptops. It lists the available laptops
            and their features including CPU, memory, storage, design and advantages
            """
        )
        return get_product_features
    except Exception as e:
        st.warning(f"Product features retriever not available: {e}")
        return None

# =====================
# PRODUCT QNA TOOLS
# =====================

product_pricing_df, product_orders_df = load_data()

@tool
def get_laptop_price(laptop_name: str) -> str:
    """
    This function returns the price of a laptop, given its name as input.
    It performs a substring match between the input name and the laptop name.
    If a match is found, it returns the price of the laptop.
    If there is NO match found, it returns -1
    """
    match_records_df = product_pricing_df[
        product_pricing_df["Name"].str.contains("^" + laptop_name, case=False)
    ]
    if len(match_records_df) == 0:
        return "-1"
    else:
        return str(match_records_df["Price"].iloc[0])

# =====================
# ORDERS AGENT TOOLS
# =====================

@tool
def get_order_details(order_id: str) -> str:
    """
    This function returns details about a laptop order, given an order ID.
    It performs an exact match between the input order id and available order ids.
    If a match is found, it returns products ordered, quantity ordered and delivery date.
    If there is NO match found, it returns -1
    """
    match_order_df = product_orders_df[
        product_orders_df["Order ID"] == order_id
    ]
    if len(match_order_df) == 0:
        return "-1"
    else:
        return str(match_order_df.iloc[0].to_dict())

@tool
def update_quantity(order_id: str, new_quantity: int) -> str:
    """
    This function updates the quantity of products (laptops) ordered for a given order Id.
    If there are no matching orders, it returns False.
    """
    match_order_df = product_orders_df[
        product_orders_df["Order ID"] == order_id
    ]
    if len(match_order_df) == 0:
        return "-1"
    else:
        product_orders_df.loc[
            product_orders_df["Order ID"] == order_id,
            "Quantity Ordered"
        ] = new_quantity
        return "True"

# =====================
# SETUP AGENTS
# =====================

@st.cache_resource
def setup_product_qa_agent():
    """Setup the Product QnA agent"""
    model, _ = setup_models()
    if model is None:
        return None
    
    system_prompt = SystemMessage("""
    You are a professional chatbot that answers questions about laptops sold by your company.
    To answer questions about laptops, you will ONLY use the available tools and NOT your own memory.
    You will handle small talk and greetings by producing professional responses.
    """)
    
    tools = [get_laptop_price]
    # Try to add product features tool if available
    features_tool = setup_product_features_retriever()
    if features_tool:
        tools.append(features_tool)
    
    checkpointer = MemorySaver()
    
    product_qa_agent = create_react_agent(
        model=model,
        tools=tools,
        state_modifier=system_prompt,
        debug=False,
        checkpointer=checkpointer
    )
    return product_qa_agent

@st.cache_resource
def setup_orders_agent():
    """Setup the Orders agent"""
    model, _ = setup_models()
    if model is None:
        return None
    
    class OrdersAgentState(TypedDict):
        messages: Annotated[list[AnyMessage], operator.add]
    
    class OrdersAgent:
        def __init__(self, model, tools, system_prompt, debug):
            self.system_prompt = system_prompt
            self.debug = debug
            
            agent_graph = StateGraph(OrdersAgentState)
            agent_graph.add_node("orders_llm", self.call_llm)
            agent_graph.add_node("orders_tools", self.call_tools)
            agent_graph.add_conditional_edges(
                "orders_llm",
                self.is_tool_call,
                {True: "orders_tools", False: END}
            )
            agent_graph.add_edge("orders_tools", "orders_llm")
            agent_graph.set_entry_point("orders_llm")
            
            self.memory = MemorySaver()
            self.agent_graph = agent_graph.compile(checkpointer=self.memory)
            
            self.tools = {tool.name: tool for tool in tools}
            self.model = model.bind_tools(tools)
        
        def call_llm(self, state: OrdersAgentState):
            messages = state["messages"]
            if self.system_prompt:
                messages = [SystemMessage(content=self.system_prompt)] + messages
            result = self.model.invoke(messages)
            return {"messages": [result]}
        
        def is_tool_call(self, state: OrdersAgentState):
            last_message = state["messages"][-1]
            return hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0
        
        def call_tools(self, state: OrdersAgentState):
            last_message = state["messages"][-1]
            tool_results = []
            for tool_call in last_message.tool_calls:
                tool = self.tools[tool_call["name"]]
                result = tool.invoke(tool_call["args"])
                tool_results.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"]
                ))
            return {"messages": tool_results}
    
    system_prompt = """
    You are a professional customer service chatbot that handles laptop orders.
    You can retrieve order details and update order quantities using the available tools.
    Always be helpful and professional in your responses.
    """
    
    tools = [get_order_details, update_quantity]
    orders_agent = OrdersAgent(model, tools, system_prompt, debug=False)
    return orders_agent

# =====================
# ROUTER AGENT
# =====================

class RouterAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class RouterAgent:
    def __init__(self, model, product_qa_agent, orders_agent, system_prompt, smalltalk_prompt, debug=False):
        self.system_prompt = system_prompt
        self.smalltalk_prompt = smalltalk_prompt
        self.model = model
        self.debug = debug
        
        router_graph = StateGraph(RouterAgentState)
        router_graph.add_node("Router", self.call_llm)
        router_graph.add_node("Product_Agent", self.product_agent_node)
        router_graph.add_node("Orders_Agent", self.orders_agent_node)
        router_graph.add_node("Small_Talk", self.respond_smalltalk)
        
        # Store agents
        self.product_qa_agent = product_qa_agent
        self.orders_agent = orders_agent
        
        router_graph.add_conditional_edges(
            "Router",
            self.find_route,
            {"PRODUCT": "Product_Agent", "ORDER": "Orders_Agent", "SMALLTALK": "Small_Talk", "END": END}
        )
        
        router_graph.add_edge("Product_Agent", END)
        router_graph.add_edge("Orders_Agent", END)
        router_graph.add_edge("Small_Talk", END)
        
        router_graph.set_entry_point("Router")
        self.router_graph = router_graph.compile()
    
    def product_agent_node(self, state: RouterAgentState):
        """Invoke product QnA agent"""
        thread_id = st.session_state.get("thread_id", str(uuid.uuid4()))
        agent_config = {"configurable": {"thread_id": thread_id}}
        result = self.product_qa_agent.invoke(state, agent_config)
        final_result = AIMessage(result['messages'][-1].content)
        return {"messages": [final_result]}
    
    def orders_agent_node(self, state: RouterAgentState):
        """Invoke orders agent"""
        thread_id = st.session_state.get("thread_id", str(uuid.uuid4()))
        agent_config = {"configurable": {"thread_id": thread_id}}
        result = self.orders_agent.agent_graph.invoke(state, agent_config)
        final_result = AIMessage(result['messages'][-1].content)
        return {"messages": [final_result]}
    
    def call_llm(self, state: RouterAgentState):
        messages = state["messages"]
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        result = self.model.invoke(messages)
        return {"messages": [result]}
    
    def respond_smalltalk(self, state: RouterAgentState):
        messages = state["messages"]
        messages = [SystemMessage(content=self.smalltalk_prompt)] + messages
        result = self.model.invoke(messages)
        return {"messages": [result]}
    
    def find_route(self, state: RouterAgentState):
        last_message = state["messages"][-1]
        destination = last_message.content
        return destination

@st.cache_resource
def setup_router_agent():
    """Setup the router agent"""
    model, _ = setup_models()
    if model is None:
        return None
    
    product_qa_agent = setup_product_qa_agent()
    orders_agent = setup_orders_agent()
    
    system_prompt = """
    You are a Router that analyzes the input query and chooses 4 options:
    SMALLTALK: If the user input is small talk, like greetings and goodbyes.
    PRODUCT: If the query is a product question about laptops, like features, specifications and pricing.
    ORDER: If the query is about orders for laptops, like order status, order details or update order quantity
    END: Default, when it's neither PRODUCT or ORDER.
    
    The output should only be just one word out of the possible 4: SMALLTALK, PRODUCT, ORDER, END.
    """
    
    smalltalk_prompt = """
    If the user request is small talk, like greetings and goodbyes, respond professionally.
    Mention that you will be able to answer questions about laptop product features and provide order status and updates.
    """
    
    router_agent = RouterAgent(model, product_qa_agent, orders_agent, system_prompt, smalltalk_prompt, debug=False)
    return router_agent

# =====================
# STREAMLIT UI
# =====================

st.set_page_config(page_title="Multi-Agent Chatbot", layout="wide")

st.title("🤖 Multi-Agent Laptop Chatbot")
st.write("Ask questions about laptop products, orders, or just chat with me!")

# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_key_valid" not in st.session_state:
    st.session_state.api_key_valid = bool(os.environ.get("AZURE_OPENAI_API_KEY"))

# Check for API key
if not st.session_state.api_key_valid:
    st.error("⚠️ Azure OpenAI API credentials not configured. Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT in Streamlit secrets.")
    st.stop()

# Setup router agent
try:
    router_agent = setup_router_agent()
    if router_agent is None:
        st.error("Failed to initialize the router agent. Check your Azure OpenAI credentials.")
        st.stop()
except Exception as e:
    st.error(f"Error setting up router agent: {e}")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about laptops, orders, or just chat..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get response from router agent
    with st.chat_message("assistant"):
        try:
            with st.spinner("Processing..."):
                # Invoke router agent
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                user_message = {"messages": [HumanMessage(prompt)]}
                
                ai_response = router_agent.router_graph.invoke(user_message, config=config)
                response_text = ai_response['messages'][-1].content
                
            st.write(response_text)
            
            # Add assistant message to history
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This multi-agent chatbot combines three specialized agents:
    - **Product QnA Agent**: Answers questions about laptop features and pricing
    - **Orders Agent**: Manages order details and updates
    - **Router Agent**: Routes your query to the appropriate agent
    """)
    
    st.divider()
    
    if st.button("🔄 Start New Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    
    st.divider()
    st.write(f"**Thread ID**: `{st.session_state.thread_id}`")
