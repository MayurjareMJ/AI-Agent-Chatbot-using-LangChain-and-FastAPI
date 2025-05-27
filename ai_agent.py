import os
GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY=os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")



from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from langchain_community.tools.tavily_search import TavilySearchResults

open_llm=ChatOpenAI(model="gpt-4o-mini")
groq_llm=ChatGroq(model="llama-3.1-8b-instant")



serch_tool=TavilySearchResults(max_results=2)

#Step3
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage
system_prompt="Act as an AI chatbot who is smart and friedly"


def get_response_from_ai_agent(llm_id,query,allow_search,system_prompt,provider):
    if provider=="groq":
        llm=ChatGroq(model=llm_id)
    elif provider=="openai":
        llm=ChatOpenAI(model=llm) 


    tool=[TavilySearchResults(max_results=2)] if allow_search else []
    agent=create_react_agent(
        llm=llm,
        tools=tool,
        state_modifier=system_prompt
    ) 

    if allow_search:
        # Use the search tool
        agent=create_react_agent(
            model=llm,
            tools=[serch_tool],
            state_modifier=system_prompt
        )
    else:
        # Use the LLM directly
        agent=create_react_agent(
            llm=open_llm,
            tools=[],
            state_modifier=system_prompt
        )
    
    # Invoke the agent with the query
    state ={"message":query}
    response=agent.invoke(state)
    return response

    agent=create_react_agent(
    llm=groq_llm,
    tools=[serch_tool],
    state_modifier=system_prompt)

    state ={"message": query}
    response=agent.invoke(state)
    message=response.get("Message")
    ai_message=[message.content for message in messages if isinstance(message,AIMessage)]
    return ai_message[-1]
















