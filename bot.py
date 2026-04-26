"""Discord bot for the RAG FAQ chatbot, with per-user conversation memory.

Slash commands:
    /ask question:<text>   -- ask a question; remembers your last few turns
    /forget                 -- clear your conversation history with the bot

Reactions: 👍 / 👎 on bot answers send feedback to the backend.

Memory model: a per-user sliding window of the last MEMORY_TURNS exchanges
(default 3, i.e. 6 messages). In-memory only — restarting the bot wipes
history. For persistence across restarts, swap `user_history` for Redis or
a SQLite table.

Run with:
    python bot.py
"""

import os
from collections import deque
from typing import Deque

import discord
import httpx
from discord import app_commands
from dotenv import load_dotenv

from logger_config import configure_logging, get_logger

load_dotenv()
configure_logging()
log = get_logger("bot")

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
HTTP_TIMEOUT = 30.0
MEMORY_TURNS = int(os.getenv("MEMORY_TURNS", 3))  # 3 exchanges = 6 messages

if not DISCORD_BOT_TOKEN:
    raise RuntimeError("DISCORD_BOT_TOKEN not set in .env")

intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True

client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

# Map message_id -> (query, answer) so reactions can be turned into feedback.
pending_feedback: dict[int, tuple[str, str]] = {}

# Per-user conversation history. Each deque holds at most 2*MEMORY_TURNS
# messages (alternating user / assistant).
user_history: dict[int, Deque[dict]] = {}


def get_history(user_id: int) -> Deque[dict]:
    """Get or create the history deque for a user."""
    if user_id not in user_history:
        user_history[user_id] = deque(maxlen=MEMORY_TURNS * 2)
    return user_history[user_id]


@client.event
async def on_ready():
    await tree.sync()
    log.info(
        "bot_ready",
        user=str(client.user),
        guilds=len(client.guilds),
        memory_turns=MEMORY_TURNS,
    )


@tree.command(name="ask", description="Ask a question about the AI Bootcamp.")
@app_commands.describe(question="What would you like to know?")
async def ask(interaction: discord.Interaction, question: str):
    user_id = interaction.user.id
    history = get_history(user_id)

    log.info(
        "ask_received",
        user=str(interaction.user),
        user_id=user_id,
        question=question,
        history_turns=len(history) // 2,
    )

    await interaction.response.defer(thinking=True)

    payload = {
        "query": question,
        "user_id": str(user_id),
        "history": list(history) if history else None,
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as http:
            resp = await http.post(
                f"{BACKEND_URL}/api/rag-query", json=payload
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        log.exception("backend_call_failed")
        await interaction.followup.send(
            f"⚠️ Sorry, I couldn't reach the knowledge base right now. ({type(e).__name__})"
        )
        return

    answer = data["answer"]
    sources = data.get("sources", [])
    latency = data.get("latency_ms", 0)

    # Update the user's history with this exchange BEFORE sending the reply,
    # so a reply that fails to send still leaves the history consistent with
    # what the LLM was actually shown.
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})

    embed = discord.Embed(
        title=f"❓ {question[:240]}",
        description=answer[:4000],
        color=0x5865F2,
    )
    if sources:
        embed.add_field(
            name="Sources",
            value=", ".join(f"`{s}`" for s in sources[:5]),
            inline=False,
        )
    memory_note = (
        f" · Memory: {len(history) // 2} turn(s)" if history else ""
    )
    embed.set_footer(
        text=f"Answered in {latency} ms · React 👍/👎 for feedback{memory_note}"
    )

    sent = await interaction.followup.send(embed=embed, wait=True)
    pending_feedback[sent.id] = (question, answer)
    try:
        await sent.add_reaction("👍")
        await sent.add_reaction("👎")
    except discord.HTTPException:
        log.warning("reaction_add_failed", message_id=sent.id)


@tree.command(
    name="forget",
    description="Clear your conversation history with the bot.",
)
async def forget(interaction: discord.Interaction):
    user_id = interaction.user.id
    cleared = user_id in user_history and len(user_history[user_id]) > 0
    user_history.pop(user_id, None)
    log.info("history_cleared", user_id=user_id, had_history=cleared)
    msg = (
        "🧠 I've forgotten our previous conversation. Fresh start!"
        if cleared
        else "Nothing to forget — we don't have any history yet."
    )
    await interaction.response.send_message(msg, ephemeral=True)


@client.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
    if payload.user_id == client.user.id:
        return
    if payload.message_id not in pending_feedback:
        return
    if str(payload.emoji) not in {"👍", "👎"}:
        return

    rating = "up" if str(payload.emoji) == "👍" else "down"
    query, answer = pending_feedback[payload.message_id]

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as http:
            await http.post(
                f"{BACKEND_URL}/api/feedback",
                json={
                    "query": query,
                    "answer": answer,
                    "rating": rating,
                    "user_id": str(payload.user_id),
                },
            )
        log.info(
            "feedback_sent",
            rating=rating,
            user_id=payload.user_id,
            message_id=payload.message_id,
        )
    except httpx.HTTPError:
        log.exception("feedback_send_failed", message_id=payload.message_id)


if __name__ == "__main__":
    client.run(DISCORD_BOT_TOKEN)
