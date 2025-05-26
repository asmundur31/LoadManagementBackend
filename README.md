# LoadManagement Backend

This is the backend for my master's thesis project, developed as an open-source repository. It includes:

1. **Backend API** – Powers the LoadManagement frontend app.
2. **Algorithm implementations** – Used to estimate training load features.
3. **TeamGym dataset** – Collected as part of the project, including:
   - Data from 9 gymnasts performing a structured protocol.
   - Data from 2 gymnasts during regular training sessions.

For a deeper understanding of the project, its context, and how the data and algorithms are used, please refer to the accompanying thesis report.

# 🚀 Getting Started: Run the Project Locally with Docker

This project is containerized using **Docker** and **Docker Compose** to make setup and development easy and consistent. Follow the steps below to get the entire app—including PostgreSQL, Redis, Celery, and your web app—running locally.

---

## 📦 Prerequisites

Before you begin, make sure you have the following installed:

- [Docker Desktop](https://www.docker.com/products/docker-desktop) (includes Docker and Docker Compose)

---

## 🧱 Step-by-Step Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/asmundur/LoadManagementBackend.git
cd LoadManagementBackend
```

### 2. Set Up Environment Variables

The project requires a .env file for service configuration (PostgreSQL, Redis, Celery, etc.).

#### 📁 Create the .env file

```bash
cp .env.example .env
```

#### ✏️ Edit .env if needed

```bash
# .env

# Database connection
DATABASE_URL=postgresql://myuser:mypassword@db:5432/mydb

# Celery and Redis
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0
```

✅ The default values should work out-of-the-box with Docker Compose.

### 3. Start All Services

This will build and start:

- The main app (api)
- The database (postgres)
- Redis server (redis)
- Celery worker (celery)

```bash
docker-compose up --build
```

⏳ On the first run, it may take a few minutes to download and build everything.

### 4. Access the Application

Once the containers are up:

- App: http://localhost:8000

### 🛑 Stopping the App

To shut everything down:

```bash
docker-compose down
```

To remove volumes (⚠️ including database data):

```bash
docker-compose down -v
```

# Upload data

To use this backend a frontend app has been developed that is available here [Load Management Frontend](https://github.com/asmundur31/LoadManagement)

# Contact

If any questions about this repository or the project please contact the creator at: asmundur31@gmail.com
