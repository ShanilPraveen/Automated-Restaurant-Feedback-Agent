import os
from dotenv import load_dotenv
import pandas as pd
import operator
from typing import TypedDict

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

from tools import (
    generate_recommendations_report, 
    plot_pie_chart, 
    plot_line_chart, 
    plot_stacked_bar_chart, 
    plot_simple_bar_chart,
    analyze_sentiment,
    save_report_as_pdf
)
from data import ReviewDatabase

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#The Tools

@tool
def get_recommendations_report(start_date:str,end_date:str)->str:
    """
    Generates a full strategic report with recommendations for a specific date range.
    The report analyzes sentiment trends and key positive/negative themes from reviews.
    
    Args:
        start_date (str): The start date for the analysis in 'YYYY-MM-DD' format.
        end_date (str): The end date for the analysis in 'YYYY-MM-DD' format.
        
    Returns:
        str: A professional-style report with strategic recommendations.
    """
    database = ReviewDatabase("European Restaurant Reviews.csv")
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    report = generate_recommendations_report(database, start_date, end_date)
    return report

@tool
def get_sentiment_visualization(chart_type: str, start_date: str, end_date: str) -> str:
    """
    Generates a visual chart showing sentiment trends over a date range.
    
    Args:
        chart_type (str): The type of chart to generate. Options are 'pie', 'line', 
                          'stacked_bar', or 'simple_bar'.
        start_date (str): The start date for the analysis in 'YYYY-MM-DD' format.
        end_date (str): The end date for the analysis in 'YYYY-MM-DD' format.
        
    Returns:
        str: The path to the saved image file.
    """
    database = ReviewDatabase('European Restaurant reviews.csv')
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    reviews_df = database.get_reviews(start_ts, end_ts)

    if chart_type == 'pie':
        return plot_pie_chart(reviews_df)
    elif chart_type == 'line':
        return plot_line_chart(reviews_df)
    elif chart_type == 'stacked_bar':
        return plot_stacked_bar_chart(reviews_df)
    elif chart_type == 'simple_bar':
        return plot_simple_bar_chart(reviews_df)
    else:
        return "Invalid chart type specified. Please choose from 'pie', 'line', 'stacked_bar', or 'simple_bar'."


@tool
def generate_feedback_response(review_text: str) -> str:
    """Generate a customer service response to a restaurant review.
    
    Args:
        review_text: The customer review text to respond to
    
    Returns:
        A polite, context-aware customer service response
    """
    sentiment = analyze_sentiment(review_text)
    response_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a customer service representative for SteamNoodles. Your task is to generate a short, polite, and context-aware response to a customer review. The review has a sentiment of: {sentiment}."),
        ("user", "Review: {review_text}\nResponse:")
    ])

    response_chain = response_prompt | ChatGroq(model="llama-3.1-8b-instant")
    result = response_chain.invoke({
        "sentiment": sentiment,
        "review_text": review_text
    })
    return result.content

@tool
def save_report_to_pdf(report: str, file_name: str) -> str:
    """Save a report as a PDF file.
    
    Args:
        report: The report content to save
        file_name: The name of the PDF file to create
        
    Returns:
        str: Confirmation message with the saved file path
    """
    return save_report_as_pdf(report, file_name)

#The Memory
class AgentState(TypedDict):
    input: str
    agent_outcome: str

#The Agent and their reasoning loops
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY,temperature=0)

#Strategic Recommendation Agent
tools_rec = [get_recommendations_report, save_report_to_pdf]
prompt_rec = ChatPromptTemplate.from_messages([
    ("system", """
    You are a strategic recommendation agent. Your task is to generate reports based on a user's request. You have two tools: `get_recommendations_report` and `save_report_to_pdf`.

    Here are your rules:
    1.  First, you must always use the `get_recommendations_report` tool to get the report's content.
    2.  After generating the report content, you MUST check if the user's original request asked to "save," "create a PDF," or "export" the report.
    3.  If the user requested to save the report, you MUST then use the `save_report_to_pdf` tool with the report content from the first step.
    4.  If the user did NOT request to save the report, you MUST simply return the content of the report to the user.
    """), 
    ("user", "{input}"), 
    ("placeholder", "{agent_scratchpad}")
])
agent_rec = create_tool_calling_agent(llm,tools_rec, prompt_rec)
agent_executor_rec = AgentExecutor(agent=agent_rec, tools=tools_rec, verbose=True)

#Sentiment Visualization Agent
tools_plot = [get_sentiment_visualization]
prompt_plot = ChatPromptTemplate.from_messages([
    ("system", """
    You are a highly skilled sentiment plotting agent. Your task is to select the most appropriate chart type to visualize a user's request, and then use the `get_sentiment_visualization` tool to generate a single chart.

    Here are the rules you must follow for chart selection:
    1.  If the user's request mentions "trend," "over time," "monthly," or "change," you must choose the 'line' chart.
    2.  If the user's request mentions "distribution," "breakdown," "percentage," or "composition," you must choose the 'pie' chart.
    3.  For any other type of visualization request, you must choose either 'stacked_bar' or 'simple_bar'.
    
    You MUST extract the start and end dates from the user's input. You MUST NOT generate more than one chart per request.
    """),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent_plot = create_tool_calling_agent(llm,tools_plot, prompt_plot)
agent_executor_plot = AgentExecutor(agent=agent_plot, tools=tools_plot, verbose=True)

#Feedback Response Agent
tools_feedback = [generate_feedback_response]
prompt_feedback = ChatPromptTemplate.from_messages([
    ("system", "You are a customer feedback agent. Your only goal is to generate a response to a single customer review using the `generate_feedback_response` tool. The final output must be only the result from the tool call, with no extra commentary or conversation."),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent_feedback = create_tool_calling_agent(llm,tools_feedback, prompt_feedback)
agent_executor_feedback = AgentExecutor(agent=agent_feedback, tools=tools_feedback, verbose=True)

#Nodes functions to execute each agent
def strategic_recommendation_node(state):
    result  = agent_executor_rec.invoke({"input": state["input"]})
    return {"agent_outcome":result["output"]}

def sentiment_plotting_node(state):
    result = agent_executor_plot.invoke({"input": state["input"]})
    return {"agent_outcome":result["output"]}

def feedback_response_node(state):
    result = agent_executor_feedback.invoke({"input": state["input"]})
    return {"agent_outcome":result["output"]}


#Orchestration
def route_request(state):
    user_input = state["input"].lower()
    if "report" in user_input or "recommendations" in user_input:
        return "strategic_recommendations"
    elif "chart" in user_input or "plot" in user_input or "visualization" in user_input:
        return "sentiment_plotting"
    elif "review" in user_input or "respond" in user_input or "feedback" in user_input:
        return "feedback_response"
    else:
        return "sentiment_plotting" 
    

workflow = StateGraph(AgentState)
workflow.add_node("router",lambda state: {"input": state["input"]})
workflow.add_node("strategic_recommendations",strategic_recommendation_node)
workflow.add_node("sentiment_plotting",sentiment_plotting_node)
workflow.add_node("feedback_response",feedback_response_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router",
    route_request,
    {
        "strategic_recommendations": "strategic_recommendations",
        "sentiment_plotting": "sentiment_plotting",
        "feedback_response": "feedback_response"
    }
)

workflow.add_edge("strategic_recommendations",END)
workflow.add_edge("sentiment_plotting",END)
workflow.add_edge("feedback_response",END)

app = workflow.compile()


