CREATE EXTENSION IF NOT EXISTS vector;

-- Create a user for the test environment
CREATE ROLE test_user WITH LOGIN PASSWORD 'test_password';
ALTER ROLE test_user CREATEDB;
