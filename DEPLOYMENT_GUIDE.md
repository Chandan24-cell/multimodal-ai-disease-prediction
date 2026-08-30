# Deployment Guide

This application has two deployable services: a FastAPI backend and a Vite frontend. The backend loads a 327 MB ViT checkpoint, so the Render free tier is not a supported production target.

## 1. Prepare the model image

The backend Docker build expects this file in the build context:

```text
backend/models/vit/medical_finetuned/model.safetensors
```

The checkpoint must be downloaded before the Docker build. The Dockerfile never downloads model weights or the Hugging Face text model during its build. Build locally or in CI, then push the finished image to Docker Hub:

```bash
docker login
docker buildx create --name healthcare-builder --use 2>/dev/null || docker buildx use healthcare-builder
docker buildx build \
  --platform linux/amd64 \
  --tag YOUR_DOCKERHUB_USER/multimodal-healthcare-backend:latest \
  --push \
  ./backend
```

Replace `YOUR_DOCKERHUB_USER` with your Docker Hub username. The final command both builds and pushes the AMD64 image, so a separate `docker push` is not required. To build and push as separate commands instead, add `--load` to the build command and run `docker push YOUR_DOCKERHUB_USER/multimodal-healthcare-backend:latest` afterward.

Using a pre-built Docker Hub image avoids Render's constrained pip build entirely. Keep the image private if the checkpoint is not intended to be public.

## 2. Configure MongoDB Atlas

1. Create an Atlas project and an application database user.
2. Create a cluster in a region close to the backend.
3. In **Network Access**, allow Render's outbound traffic. For an initial test, Atlas supports `0.0.0.0/0`; restrict this to known egress IPs before production.
4. Copy the connection string and URL-encode special characters in the username or password.
5. Use database name `multimodal_healthcare` in the URI or set `MONGODB_DB_NAME` separately.

## 3. Deploy the backend on Render

Recommended options:

- **Recommended:** Render Starter, currently $7/month, or a larger instance if the model and workload require it. ML inference should not run on the free 512 MB instance.
- **Build avoidance:** deploy the Docker Hub image created above, so Render pulls instead of builds it. The runtime still needs enough memory for PyTorch and the ViT model.

For a source Docker deployment, create a Render Web Service with:

- Root directory: repository root
- Dockerfile path: `backend/Dockerfile`
- Health check path: `/api/health`
- Start command: use the Dockerfile command, with one worker

For a pre-built image deployment, select the image from Docker Hub and expose port `8000`.

Set these Render environment variables:

```text
MONGODB_URI=mongodb+srv://USER:PASSWORD@CLUSTER.mongodb.net/multimodal_healthcare?retryWrites=true&w=majority
MONGODB_DB_NAME=multimodal_healthcare
JWT_SECRET_KEY=<output of: openssl rand -base64 48>
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
FRONTEND_URL=https://YOUR_FRONTEND.vercel.app
APP_ENV=production
```

Do not commit `.env` files or place credentials in the Docker image.

## 4. Bootstrap the first Admin

After the backend can reach Atlas, run the script from a trusted machine using the same MongoDB variables:

```bash
export MONGODB_URI='mongodb+srv://...'
export MONGODB_DB_NAME='multimodal_healthcare'
python scripts/initialize_admin.py \
  --username admin \
  --email admin@example.com \
  --full-name 'System Administrator'
```

It prompts for the password without echoing it and refuses to run if any user already exists. The public registration endpoint also assigns the first user `Admin` and every later user `Staff`.

## 5. Deploy the frontend on Vercel

1. Import the repository into Vercel.
2. Set the project root to `frontend` if Vercel does not detect it automatically.
3. Build command: `npm run build`.
4. Output directory: `dist`.
5. Add this environment variable for **Production**, **Preview**, and **Development** as needed:

```text
VITE_API_URL=https://YOUR_BACKEND.onrender.com/api
```

Redeploy after changing a `VITE_` variable because Vite embeds it at build time. The frontend falls back to `/api` only when it is served by the FastAPI backend itself.

## 6. Verify the deployment

Check the backend first:

```bash
curl -i https://YOUR_BACKEND.onrender.com/api/health
```

A connected deployment returns JSON containing `"status": "healthy"` and `"database": "connected"`. A reachable deployment with a failed Atlas connection returns `"status": "degraded"` and logs the connection error.

Then test login with form-encoded credentials:

```bash
curl -i -X POST https://YOUR_BACKEND.onrender.com/api/auth/login \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  --data-urlencode 'username=admin' \
  --data-urlencode 'password=YOUR_PASSWORD'
```

If login fails, inspect Render logs for the username lookup, MongoDB availability, and HTTP status. Password values are never logged. In the browser, verify the request URL is the Render URL plus `/api/auth/login`, not a Vercel-relative URL.

## Local production-like Compose

For the two-container local integration deployment, the frontend is published
on port 80 and calls the backend at `http://localhost:8000`. Use a local `.env`
with MongoDB credentials and run this single command from the repository root:

```bash
docker compose down --volumes --remove-orphans && docker compose up --build -d
```

The `down --volumes` portion removes the old containers and the local MongoDB
volume, so it also removes local database data. Omit `--volumes` when that data
must be preserved:

```bash
docker compose down --remove-orphans && docker compose up --build -d
```

The default Compose file runs the baked backend image with one Uvicorn worker.
Because the local MongoDB service enables authentication, Compose injects the
credentialed internal URI using `mongodb:27017` and `authSource=admin`. The
frontend build receives `VITE_API_URL=http://localhost:8000` automatically.

Open these URLs after the command completes:

- Frontend application: `http://localhost`
- Backend health check: `http://localhost:8000/api/health`
- Backend API documentation: `http://localhost:8000/api/docs`

Check service status and logs without opening another terminal:

```bash
docker compose ps
docker compose logs --tail=100 backend frontend
```

## Run the unified local app without Docker

On macOS or Linux, the repository root also includes `run_local.sh`. It creates
the virtual environment if needed, installs backend dependencies, builds the
React app into `frontend/dist`, starts the local MongoDB Compose service, and
starts FastAPI. Run:

```bash
chmod +x run_local.sh
./run_local.sh
```

Open `http://localhost:8000`. The backend serves the React build at that same
URL and keeps API routes under `/api/`. Stop it with `Ctrl+C`.

The local runner intentionally does not use Uvicorn `--reload`: the ViT model
is large enough that the reload watcher can temporarily load multiple model
processes and trigger exit code 137 from memory pressure. Restart the script
manually after backend code changes.

The script uses `docker-compose.host.yml` to publish MongoDB only on
`127.0.0.1:27017`, which is needed because FastAPI runs directly on the host.
The standard `docker-compose.yml` keeps MongoDB unexposed when all services run
in containers.
