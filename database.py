

import sqlite3

conn = sqlite3.connect("databases/support_tickets.db")
cursor = conn.cursor()


ALLOWED_ACTIONS = [
    "create_ticket",
    "update_ticket",
    "add_step",
    "close_ticket"
]


