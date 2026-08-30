"""Root-level ASGI entrypoint for uvicorn when launched from the repository root."""

from backend.main import app

# Export the same app object expected by: uvicorn main:app
# This allows the root command to work without changing directories.

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
