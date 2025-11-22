import datetime
from google.adk.agents.llm_agent import Agent, LlmAgent
from google.adk.models.lite_llm import LiteLlm

import yfinance as yf

def get_current_time() -> dict:
  """Returns the current time.
  Returns:
    dict: status and result or error msg.
  """

  now = datetime.datetime.now()
  report = (f'The current time is {now.strftime("%H:%M:%S")}')
  return {"status": "success", "report": report}



def get_stock_price(ticker: str):
  """
  Fetch the current stock price for a given ticker symbol.

  Args:
    ticker (str): The stock ticker symbol.

    Returns:
      float: The current stock price.
  """
  stock = yf.Ticker(ticker)
  price = stock.info.get("currentPrice", "Price not available")
  return {"price": price, "ticker": ticker}


mymodel = LiteLlm(
  api_base='http://localhost:11434/v1',
  model='openai/llama3.2:latest',
  api_key='ollama'
)

time_agent = Agent(
  name="time_agent",
  description="A helpful agent that provides the current time.",
  instruction="You are a time assistant. Always use the get_current_time tool.",
  model=LiteLlm(
    api_base='http://localhost:11434/v1',
    model='openai/llama3.2:latest',
    api_key='ollama'
  ),
  tools=[get_current_time],
)

base_agent = LlmAgent(
  name="stock_price_agent",
  description="A helpful agent that gets stock price.",
  instruction=(
    "You are a stock price assistant. Always use the get_stock_price tool."
    "Include the ticker symbol in your response."
    "You have access to a specialist sub-agent called `time_agent` for time-related queries."
  ),
  model=mymodel, #LiteLlm(model="ollama_chat/llama3.2:latest"),
  tools=[get_stock_price],
  sub_agents=[time_agent],
)

root_agent = base_agent
