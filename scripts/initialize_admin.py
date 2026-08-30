#!/usr/bin/env python3
"""Create the first Admin user in MongoDB without exposing the password in logs."""
import argparse
import asyncio
import getpass
import os
import sys

from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


async def initialize_admin(username: str, email: str, full_name: str, password: str) -> None:
    mongodb_uri = os.getenv("MONGODB_URI")
    database_name = os.getenv("MONGODB_DB_NAME", "multimodal_healthcare")
    if not mongodb_uri:
        raise RuntimeError("MONGODB_URI must be set before running this script")

    client = AsyncIOMotorClient(mongodb_uri, serverSelectionTimeoutMS=5000)
    try:
        await client.admin.command("ping")
        users = client[database_name].users
        if await users.count_documents({}) > 0:
            raise RuntimeError("Users already exist; refusing to create or replace an Admin")

        result = await users.insert_one({
            "username": username,
            "email": email,
            "full_name": full_name,
            "hashed_password": pwd_context.hash(password),
            "role": "Admin",
            "is_active": True,
        })
        print(f"Created first Admin user with id {result.inserted_id}")
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap the first Admin user")
    parser.add_argument("--username", required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument("--full-name", required=True)
    args = parser.parse_args()
    password = getpass.getpass("Admin password: ")
    if len(password) < 8:
        parser.error("Admin password must contain at least 8 characters")

    try:
        asyncio.run(initialize_admin(args.username, args.email, args.full_name, password))
    except Exception as exc:
        print(f"Admin initialization failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
