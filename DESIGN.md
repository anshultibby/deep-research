# Deep Research Agent - Design Document

## Overall Architecture

Straightforward toolcalling agent with a few tools to perform websearch in a systematic way.

tools:
1. Search - uses serper API to search on google and return URLs, titles and content which is scraped using the helper function for scraping
2. Modify_checklist - main planning tool which allows agent to write or update a research checklist item
3. Get_currrent_checklist - Gets the state of the current checklist as pending/completed items
4. Write subreport - note taking tool which allows an agent to fulfil individual checklist items. Can support with citations.
5. Write final report - write final report tool which synthesizes all the data and puts the citations together. Also ends the research call.

### System architecture description
Built a very simple backend in fastapi and a frontend using nextjs to try out the agent. I have a streaming system using SSE to emit tool calls and checklist items as the LLM makes them. I am storing the entire chat in the backend as a json so that I can review the flow and figure out if behaviour matches what I expect (didn't use Langgraph and more sophisticated tracing methods just because of time constraints but I have tried it in past)


## Activity log:
a1. I started with my basic agent design to see if it will be good enough. This is pretty close to the starting design and I am happy with it! My first time trying out langchain stategraph and it is pretty clean api.
a2. For tools I have a notion of context variables, I think in langchain these are just like injected_vars for the tool. I am a fan of these as they allow me to send info such as chat_id, user_id through code without needing to involve the LLM.
a3. I truncated the content that is scraped and also the scraper is very rudimentary just using bs4 so the quality of the final output may not be so high. I can increase these limits for higher quality answers.

b. I have an extra tool call ask_clarification but it was gonna take too long to stream the intermediate response to the user and wait for their input so I removed it from final version. I guess this means that I didn't build a chat agent just a single shot deepresearch agent. This could be a big shortcoming but it is not at all difficult to convert this into a chat agent by just adding the user message to conversation list and then sending it to the llm again.

c. Most things worked pretty well from my inital design. It was a little bit tricky to get streaming working but I looked through langchain's repo and found the SSE style architecture very neat so I just copied it using cursor for my agent.

## Shortcomings:
1. Scraping is rudimentary - can run into bot detection sort of issues and no intelligent parsing of content. Can solve by using a better API which supports all this from out of the box
2. Agent research loop is pretty short. It can use more reflection steps and propmt to guide it and help it decide when to add more checklist items after going through an initial set of them vs calling it a day.
3. All sources just exist in chat context. This can easily cause bloat. Would be better to implement the resource idea from MCP so that we can pull in results and then look through them systematically via grep and reading chunks.
4. The tool implementation is a bit bespoke, I use the langchain tool prior but modify it slightly. I am also not sure about the relationship between MCP and Langchain's priors.


## Things I will add:
1. Evals on a widesearch dataset
2. A tool for scraping content from a specific url. Usecase is to allow LLM to step into some of the inital webpages it finds using Serper.
3. Coding capabilities. I will add a couple coding tools (write file, run code) and a virtualized file system. Coding is a very effective way for LLM to do analysis and LLMs are great at writing code. Also Anthropic seems to recommend code first methodology so as to improve security of the application and reduce token usage
4. Full chat agent system
5. MCP support? nice to have for user to add any mcp that they may want easily
6. Multi agent architecture? - One way to implement is to think of our main chat agent as the orchestrator. Each time it creates a checklist item it can create a subagent to address that checklist item. Main agent can use MCP resources to work with outputs from individual subagents. This can immediately increase the complexity of search plan and hopefully let us tackle topics that are too large to fit into a single agent's context.
