FileIntel  has Docker support. Instructions for deployment:

  Step 1: Build the Docker Image

  # Navigate to the FileIntel directory
  cd /mnt/c/code/FileIntel

  # Build the Docker image with a specific tag
  docker build -t fileintel:latest .

  Step 2: Save the Image as a TAR File

  # Save the image to a tar file
  docker save -o fileintel-image.tar fileintel:latest

  # Optionally compress it to reduce file size
  gzip fileintel-image.tar
  # This creates fileintel-image.tar.gz

  Step 3: Transfer and Load on Target Server

  On your target server:

  # If you compressed the file, decompress it first
  gunzip fileintel-image.tar.gz

  # Load the image into Docker
  docker load -i fileintel-image.tar

  # Verify the image was loaded
  docker images | grep fileintel

  Step 4: Create Production Docker Compose

  You'll also need to transfer your configuration. Let me create a production-ready docker-compose file:

● Write(docker-compose.prod.yml)
  ⎿  Wrote 92 lines to docker-compose.prod.yml
     services:
       postgres:
         image: "pgvector/pgvector:pg13"
         environment:
           POSTGRES_USER: ${POSTGRES_USER:-fileintel_user}
           POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-your_secure_password}
           POSTGRES_DB: ${POSTGRES_DB:-fileintel}
         ports:
           - "${POSTGRES_PORT:-5432}:5432"
         volumes:
     … +82 lines (ctrl+r to expand)

● Write(.env.example)
  ⎿  Wrote 20 lines to .env.example
     # Database Configuration
     POSTGRES_USER=fileintel_user
     POSTGRES_PASSWORD=your_secure_password_here
     POSTGRES_DB=fileintel

     # API Configuration (only port that needs external access)
     API_PORT=8000

     # OpenAI Configuration (required)
     OPENAI_API_KEY=your_openai_api_key_here

     # Optional: Override default paths
     # UPLOADS_DIR=/custom/uploads
     # PROMPTS_DIR=/custom/prompts
     # INPUT_DIR=/custom/input
     # OUTPUT_DIR=/custom/output

● Step 5: Complete Deployment Commands

  Here's the complete process:

  On Your Development Machine:

  # Build and save the image
  cd /mnt/c/code/FileIntel
  docker build -t fileintel:latest .
  docker save -o fileintel-image.tar fileintel:latest
  gzip fileintel-image.tar

  # Create deployment package
  tar -czf fileintel-deploy.tar.gz \
    fileintel-image.tar.gz \
    docker-compose.prod.yml \
    .env.example \
    prompts/ \
    config/ \
    run_worker.py

  On Your Target Server:

  # Extract deployment package
  tar -xzf fileintel-deploy.tar.gz

  # Load Docker image
  gunzip fileintel-image.tar.gz
  docker load -i fileintel-image.tar

  # Create environment file
  cp .env.example .env
  # Edit .env with your actual values (OPENAI_API_KEY, passwords, etc.)
  nano .env

  # Create required directories
  mkdir -p logs uploads input output

  # Deploy with Docker Compose
  docker-compose -f docker-compose.prod.yml up -d

  # Check services
  docker-compose -f docker-compose.prod.yml ps

  Step 6: Portainer Deployment

  For Portainer deployment:

  1. Upload the image to Portainer:
    - Go to Portainer → Images → Import
    - Upload the fileintel-image.tar file
  2. Create a new stack:
    - Go to Portainer → Stacks → Add Stack
    - Copy the contents of docker-compose.prod.yml
    - Set environment variables or upload your .env file
  3. Deploy the stack

● The deployment package is ready! You now have:

  - Production Docker Compose (docker-compose.prod.yml)
  - Environment template (.env.example)
  - Complete deployment commands above

  The image will include all your recent fixes (position field for chunk ordering and improved citation chunking). Just remember to
  set your OPENAI_API_KEY in the .env file on the target server.
