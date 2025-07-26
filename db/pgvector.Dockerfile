FROM pgvector/postgres:latest

# Default database variables (overridden by docker-compose)
ENV POSTGRES_USER=rag_user
ENV POSTGRES_PASSWORD=rag_password
ENV POSTGRES_DB=rag

# Copy initialisation script into the image
COPY init.sql /docker-entrypoint-initdb.d/