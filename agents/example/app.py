from a2a.server.apps import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import AgentCard

# 4. Agent Identity (The Discovery Card)
agent_card = AgentCard(
    name="InfraDesignAgent",
    description="Infrastructure Agent with remote MCP tool access.",
    capabilities={"streaming": True}
)

# 5. Initialize the A2A Server
class LangGraphExecutor:
    def __init__(self, graph):
        self.graph = graph

    async def execute(self, context, event_queue):
        user_msg = context.get_user_input()
        # Use thread_id for Cosmos DB persistence
        config = {"configurable": {"thread_id": context.task_id}}
        
        async for chunk in self.graph.astream({"messages": [user_msg]}, config):
            await event_queue.add_text_part(str(chunk))
        await event_queue.complete()

# FastAPI App Export
app = A2AFastAPIApplication(
    agent_card=agent_card,
    request_handler=DefaultRequestHandler(
        agent_executor=LangGraphExecutor(await create_agent())
    )
)
