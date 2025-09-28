-- Initialize test database with required extensions and basic setup
-- This script runs when the test PostgreSQL container starts

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- Create test schemas if needed
CREATE SCHEMA IF NOT EXISTS public;

-- Grant permissions to test user
GRANT ALL PRIVILEGES ON DATABASE fileintel_test TO test;
GRANT ALL ON SCHEMA public TO test;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO test;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO test;

-- Create basic test configuration
INSERT INTO public.test_config (key, value) VALUES
    ('test_env', 'docker'),
    ('initialized_at', NOW()::text)
ON CONFLICT (key) DO UPDATE SET
    value = EXCLUDED.value,
    updated_at = NOW();

-- Ensure test user can create tables and extensions
ALTER USER test CREATEDB;
GRANT CREATE ON SCHEMA public TO test;
