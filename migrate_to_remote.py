"""One-time: copy the data on this laptop into the remote database.

Render's container starts blank, so before the first deploy the rows that are
already on the laptop have to travel to the remote database. Run it once from
the project folder:

    python migrate_to_remote.py

DB_URL and DB_AUTH_TOKEN come from the environment or from .env. The script
refuses to write into a destination that already holds data, unless you pass
--force, because that would overwrite whatever the server has collected so far.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from dotenv import load_dotenv

import database

load_dotenv()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="source", default="data/edu_bot.db",
                        help="the local database to read (default: data/edu_bot.db)")
    parser.add_argument("--force", action="store_true",
                        help="copy even if the destination already has data")
    args = parser.parse_args()

    remote = database.database_url()
    if not database.is_remote(remote):
        print("DB_URL is not set to a remote database, so there is nothing to migrate to.")
        print("Set DB_URL=libsql://... (and DB_AUTH_TOKEN) in .env and run this again.")
        return 1

    await database.init_db(remote)
    existing = await database.count_contributions(remote)
    if existing and not args.force:
        print(f"The destination already holds {existing} contribution(s).")
        print("Nothing was written. Use --force if you really mean to overwrite it.")
        return 1

    # Records written before the images table existed only point at a file on
    # disk, so store those bytes first or the copied rows would arrive broken.
    for name in await database.backfill_images_from_disk(args.source):
        print(f"Stowed the existing image file {name} into the local database.")

    moved = await database.migrate_database(args.source, remote)
    print(f"Copied from {args.source} into {remote.split('@')[-1]}:")
    for table, count in moved.items():
        print(f"  {table:<16} {count}")
    print(f"Total contributions now in the remote database: "
          f"{await database.count_contributions(remote)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
