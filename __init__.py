"""Top-level package for image_embeddings."""

import os
import sqlite3
from aiohttp import web
from server import PromptServer

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    
]

__author__ = """Comfyui-Image-Embeddings"""
__email__ = "baijunty@163.com"
__version__ = "0.0.1"

from .src.image_embeddings.nodes import NODE_CLASS_MAPPINGS
from .src.image_embeddings.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

DB_PATH = os.path.join(os.path.dirname(__file__), "animadex.db")

async def character_search_handler(request):
    query = request.query.get("query", "").strip()
    if not query:
        return web.json_response([])
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name, cn_name FROM characters WHERE name LIKE ? OR cn_name LIKE ? LIMIT 10",
            (f"%{query}%", f"%{query}%")
        )
        results = [{"name": row[0], "cn_name": row[1]} for row in cursor.fetchall()]
        conn.close()
        return web.json_response(results)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

async def artist_search_handler(request):
    query = request.query.get("query", "").strip()
    if not query:
        return web.json_response([])
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name, cn_name, trigger FROM artists WHERE name LIKE ? OR cn_name LIKE ? LIMIT 10",
            (f"%{query}%", f"%{query}%")
        )
        results = [{"name": row[0], "cn_name": row[1], "trigger": row[2]} for row in cursor.fetchall()]
        conn.close()
        return web.json_response(results)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

if hasattr(PromptServer, "instance"):
    PromptServer.instance.app.add_routes([
        web.get("/image_embeddings/search_characters", character_search_handler),
        web.get("/image_embeddings/search_artists", artist_search_handler),
    ])



