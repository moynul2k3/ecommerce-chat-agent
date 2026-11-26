# ecommerce-chat-agent

An embeddable FastAPI router, WebSocket connection manager, and tiered session orchestrator for e-commerce conversational support.

## Plan Tiers

1. **Free Tier (`PlanTier.FREE`)**
   - Pure AI chat agent mode with OpenAI / LangChain tool calling.
   - Built-in tool schemas for `search_products(query)` and `get_order_status(order_id)` backed by your async callbacks.
   - Zero external state dependencies (in-memory session store).

2. **Pro Tier (`PlanTier.PRO`)**
   - Everything in Free Tier.
   - Real-time human handoff with Redis-backed session state (`ai` vs `human` mode, `assigned_agent_id`, conversation history).
   - Automatic WebSocket routing: customer messages route to the assigned human support agent room when `mode == "human"`.
   - Bidirectional mode switching: either customer or agent (via commands like `/human`, `/ai`, or REST endpoints) can transition between `ai` and `human` modes.

3. **Enterprise Tier (`PlanTier.ENTERPRISE`)**
   - Everything in Pro Tier.
   - Strict multi-vendor isolation: every WebSocket connection, Redis session key, WebSocket room, and tool callback is scoped to a required `vendor_id`.
   - Dynamic per-vendor catalog and order lookup callbacks (`vendor_search_products(vendor_id, query)`, `vendor_get_order_status(vendor_id, order_id)`).

## Installation

```bash
pip install ecommerce-chat-agent
# Optional LangChain support:
pip install "ecommerce-chat-agent[langchain]"
```

## Quickstart

### Free Tier

```python
from fastapi import FastAPI
from ecommerce_chat import ChatConfig, PlanTier, create_chat_router

async def search_products(query: str):
    return [{"id": "sku-1", "name": "Wireless Headphones", "price": 99.99}]

async def get_order_status(order_id: str):
    return {"order_id": order_id, "status": "shipped"}

config = ChatConfig(
    tier=PlanTier.FREE,
    openai_api_key="sk-...",
    search_products=search_products,
    get_order_status=get_order_status,
)

app = FastAPI()
app.include_router(create_chat_router(config), prefix="/chat")
```

### Pro Tier (Redis + Human Handoff)

```python
config = ChatConfig(
    tier=PlanTier.PRO,
    redis_url="redis://localhost:6379/0",
    openai_api_key="sk-...",
    search_products=search_products,
    get_order_status=get_order_status,
)
app.include_router(create_chat_router(config), prefix="/chat")
```

### Enterprise Tier (Multi-Vendor Isolation)

```python
async def vendor_search_products(vendor_id: str, query: str):
    return await db.products.search(vendor_id=vendor_id, query=query)

async def vendor_get_order_status(vendor_id: str, order_id: str):
    return await db.orders.get(vendor_id=vendor_id, order_id=order_id)

config = ChatConfig(
    tier=PlanTier.ENTERPRISE,
    redis_url="redis://localhost:6379/0",
    openai_api_key="sk-...",
    vendor_search_products=vendor_search_products,
    vendor_get_order_status=vendor_get_order_status,
)
app.include_router(create_chat_router(config), prefix="/chat")
```

## Endpoints

When mounted at `/chat`:

- `WS /chat/ws/chat/{session_id}?vendor_id=...` — Customer WebSocket endpoint.
- `WS /chat/ws/agent/{session_id}?agent_id=...&vendor_id=...` — Human support agent WebSocket endpoint (Pro & Enterprise).
- `GET /chat/sessions/{session_id}?vendor_id=...` — Fetch session state and message history.
- `POST /chat/sessions/{session_id}/handoff?vendor_id=...` — Escalate session to `human` mode (Pro & Enterprise).
- `POST /chat/sessions/{session_id}/return-to-ai?vendor_id=...` — Return session to `ai` mode (Pro & Enterprise).

## Development & Testing

```bash
pip install -e ".[dev]"
pytest
```
