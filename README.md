# HTMX Counter App

A simple counter application built with FastAPI, HTMX, Pydantic, and Neon Serverless DB.

## Features

- Increment, decrement, and reset counter functionality
- HTMX for seamless client-server interactions without JavaScript
- Pydantic for data validation
- SQLAlchemy ORM for database interactions
- Neon Serverless PostgreSQL database for persistence
- Tailwind CSS for modern, responsive styling
- Automatic fallback to SQLite database if Neon DB connection fails

## Setup and Installation

This project uses `uv` for package management. Make sure you have `uv` installed before proceeding.

### Installing uv

If you don't have `uv` installed, you can install it with:

```bash
curl -sSf https://install.python-uv.org | python3
```

### Installing Dependencies

Install the project dependencies with:

```bash
uv pip install -e .
```

### Neon Database Configuration

1. Sign up for a free Neon account at [https://neon.tech](https://neon.tech)
2. Create a new project and database
3. Get your connection string from the Neon dashboard
4. Update the `.env` file with your connection string:

```
DATABASE_URL=postgres://username:password@endpoint/dbname
```

## Running the Application

Start the FastAPI server with:

```bash
uvicorn main:app --reload --port 3000
```

The server will run at http://127.0.0.1:3000 by default.

## How It Works

- The application uses HTMX to make requests to the server without full page reloads
- When you click increment, decrement, or reset, HTMX sends a POST request to the server
- The server updates the counter value in the Neon database
- The server returns only the HTML fragment that needs to be updated
- HTMX replaces just that part of the page with the updated content

## Project Structure

```
.
├── app/
│   ├── api/
│   │   ├── counter.py   # Counter API routes
│   │   └── routes.py    # Main application routes
│   ├── db/
│   │   ├── database.py  # Database connection setup
│   │   ├── models.py    # SQLAlchemy ORM models
│   │   ├── repository.py # Database operations
│   │   └── schemas.py   # Pydantic schemas
├── templates/
│   ├── base.html       # Base template with Tailwind CSS
│   ├── counter.html    # Counter component template
│   └── index.html      # Main page template
├── main.py             # FastAPI application entry point
├── pyproject.toml      # Project dependencies
└── .env                # Environment variables (DATABASE_URL)
```

## Development

To add more dependencies to the project:

```bash
uv pip install some-package
```

And update the pyproject.toml with the new dependency.
