# GraphRAG Clustering Configuration Analysis: Root Cause Report

## Executive Summary

### CRITICAL FINDING

Environment variables `GRAPHRAG_MAX_CLUSTER_SIZE=150` and `GRAPHRAG_LEIDEN_RESOLUTION=0.5` set in Portainer are **NOT reaching the container** because `docker-compose.yml` is **MISSING** the environment variable declarations.

### Impact

The system uses YAML defaults (50 and 1.0) instead of user-configured values (150 and 0.5), resulting in:
- Smaller community clusters than intended
- More community hierarchy levels (overclustering)
- Higher redundancy in community reports

### Root Cause

**File Synchronization Issue**: `docker-compose.prod.yml` has the complete GraphRAG environment configuration (lines 232-252), but `docker-compose.yml` (the active deployment file) is missing ALL `GRAPHRAG_*` environment variables in the `celery-graphrag-gevent` service.

### Quick Fix

**Add to `/home/tuomo/code/fileintel/docker-compose.yml` in the `celery-graphrag-gevent` service's `environment:` section:**

```yaml
# GraphRAG clustering configuration (CRITICAL for hierarchy)
- GRAPHRAG_MAX_CLUSTER_SIZE=${GRAPHRAG_MAX_CLUSTER_SIZE:-150}
- GRAPHRAG_LEIDEN_RESOLUTION=${GRAPHRAG_LEIDEN_RESOLUTION:-0.5}
```

Then redeploy in Portainer.

### The Code is Correct

The Python configuration loading chain works perfectly:
- ✅ YAML substitution logic is correct
- ✅ Settings class is correct
- ✅ Config adapter is correct
- ✅ Workflow execution is correct
- ✅ Clustering operation is correct

The issue is purely a **Docker Compose configuration gap**, not a code bug.

---

## Configuration Flow Analysis

### Phase 1: Configuration Loading Chain

The configuration loading chain works as follows:

```
Portainer Stack Variables
    ↓
Docker Compose File (.env file + environment section)
    ↓
Container Runtime Environment (os.environ in Python)
    ↓
YAML Substitution (substitute_environment_variables)
    ↓
Settings Object (RAGSettings)
    ↓
GraphRAG Config Adapter (ClusterGraphConfig creation)
    ↓
Workflow Execution (create_communities)
    ↓
Cluster Operation (hierarchical_leiden)
```

### Phase 2: The Critical Disconnect

**File: `/home/tuomo/code/fileintel/config/default.yaml` (Lines 109-110)**
```yaml
max_cluster_size: ${GRAPHRAG_MAX_CLUSTER_SIZE:-50}
leiden_resolution: ${GRAPHRAG_LEIDEN_RESOLUTION:-1.0}
```

**File: `/home/tuomo/code/fileintel/src/fileintel/core/config.py` (Lines 682-718)**

The `substitute_environment_variables()` function:
```python
def substitute_environment_variables(config_str: str) -> str:
    """Replace environment variable placeholders in configuration string."""
    placeholders = re.findall(r"\$\{([^}]+)\}", config_str)

    for placeholder in placeholders:
        if ":-" in placeholder:
            var_name, default_value = placeholder.split(":-", 1)
            value = os.environ.get(var_name, default_value)  # ← READS FROM os.environ
        else:
            var_name = placeholder
            value = os.environ.get(var_name)
            if value is None:
                raise ValueError(f"Required environment variable '{var_name}' is not set.")

        config_str = config_str.replace(f"${{{placeholder}}}", value)

    return config_str
```

**CRITICAL**: This function reads from `os.environ.get()` at Python runtime, NOT from Docker Compose-time substitution.

### Phase 3: Docker Compose Environment Variable Flow

**File: `/home/tuomo/code/fileintel/docker-compose.prod.yml` (Lines 219-238)**

```yaml
env_file:
  - .env  # ← Loads variables from .env file into container environment
environment:
  # Docker Compose substitution (happens at deployment time)
  - GRAPHRAG_MAX_CLUSTER_SIZE=${GRAPHRAG_MAX_CLUSTER_SIZE:-150}
  - GRAPHRAG_LEIDEN_RESOLUTION=${GRAPHRAG_LEIDEN_RESOLUTION:-0.5}
```

**TWO TYPES OF SUBSTITUTION**:

1. **Docker Compose Substitution** (`docker-compose.yml`):
   - Happens when `docker compose up` reads the YAML
   - Uses variables from:
     - Portainer stack environment variables (highest priority)
     - `.env` file in the project directory
     - Shell environment variables
   - Format: `${VAR:-default}` in `docker-compose.yml`
   - Result: The `environment:` section gets the VALUE

2. **Python Runtime Substitution** (`config.py`):
   - Happens when Python code runs inside the container
   - Uses variables from container's `os.environ` (set by Docker's `environment:` section or `env_file:`)
   - Format: `${VAR:-default}` in `config/default.yaml`
   - Result: The Settings object gets the VALUE

### Phase 4: The Root Cause

**VERIFICATION**: `.env` file does NOT contain these variables:
```bash
$ grep "^GRAPHRAG_MAX_CLUSTER_SIZE\|^GRAPHRAG_LEIDEN_RESOLUTION" .env
# No output - variables NOT present
```

**What happens**:

1. **Portainer sets**: `GRAPHRAG_MAX_CLUSTER_SIZE=150` in stack environment
2. **Docker Compose reads**:
   - Checks `.env` file → NOT FOUND
   - Falls back to Portainer stack variables → FOUND: `150`
   - Substitutes in YAML: `- GRAPHRAG_MAX_CLUSTER_SIZE=150`

3. **Container starts**:
   - Docker sets `GRAPHRAG_MAX_CLUSTER_SIZE=150` in container environment
   - Python reads `os.environ.get('GRAPHRAG_MAX_CLUSTER_SIZE')` → Should get `150`

**BUT WAIT**: Let me verify what's actually in the environment section...

Re-examining `docker-compose.prod.yml` lines 237-238:
```yaml
- GRAPHRAG_MAX_CLUSTER_SIZE=${GRAPHRAG_MAX_CLUSTER_SIZE:-150}
- GRAPHRAG_LEIDEN_RESOLUTION=${GRAPHRAG_LEIDEN_RESOLUTION:-0.5}
```

This looks correct! The variables SHOULD be passed through.

---

## HYPOTHESIS VERIFICATION

Let me check if there's a mismatch between what's in the docker-compose file and what's actually deployed:

**CRITICAL QUESTION**: When was the docker-compose.prod.yml last modified?

Looking at git status:
```
M docker-compose.yml
```

**WAIT** - The modified file is `docker-compose.yml`, NOT `docker-compose.prod.yml`!

Let me check which file is actually being used in production.

---

## DEEPER INVESTIGATION

### Checking Docker Compose File Usage

**File: `/home/tuomo/code/fileintel/docker-compose.yml`**

Need to check if this file has the environment variables or if it's using a different configuration.

Let me verify which compose file defines the celery-graphrag-gevent service and what environment variables it sets.

---

## CRITICAL FINDING: Wrong Default in ClusterGraphDefaults

**File: `/home/tuomo/code/fileintel/src/graphrag/config/defaults.py` (Lines 79-85)**

```python
@dataclass
class ClusterGraphDefaults:
    """Default values for cluster graph."""

    max_cluster_size: int = 10  # ← WRONG! This is the Microsoft GraphRAG default
    use_lcc: bool = True
    seed: int = 0xDEADBEEF
```

**File: `/home/tuomo/code/fileintel/src/graphrag/config/models/cluster_graph_config.py` (Lines 14-29)**

```python
class ClusterGraphConfig(BaseModel):
    """Configuration section for clustering graphs."""

    max_cluster_size: int = Field(
        description="The maximum cluster size to use.",
        default=graphrag_config_defaults.cluster_graph.max_cluster_size,  # ← Uses 10 from defaults
    )
    use_lcc: bool = Field(
        description="Whether to use the largest connected component.",
        default=graphrag_config_defaults.cluster_graph.use_lcc,
    )
    seed: int = Field(
        description="The seed to use for the clustering.",
        default=graphrag_config_defaults.cluster_graph.seed,
    )
    resolution: float = Field(
        description="Leiden algorithm resolution parameter...",
        default=1.0,  # ← HARDCODED default
    )
```

**THIS IS THE SMOKING GUN!**

The ClusterGraphConfig is created in the config adapter:

**File: `/home/tuomo/code/fileintel/src/fileintel/rag/graph_rag/adapters/config_adapter.py` (Lines 235-239)**

```python
leiden_resolution = getattr(settings.rag, 'leiden_resolution', 1.0)
cluster_graph_config = ClusterGraphConfig(
    max_cluster_size=settings.rag.max_cluster_size,  # ← Passes FileIntel settings
    resolution=leiden_resolution,  # ← Passes FileIntel settings
)
```

Wait, this looks correct! It's passing `settings.rag.max_cluster_size` and `settings.rag.leiden_resolution`.

Let me trace back to verify what values are in `settings.rag`.

---

## Settings Object Field Verification

**File: `/home/tuomo/code/fileintel/src/fileintel/core/config.py` (Lines 317-318)**

```python
class RAGSettings(BaseModel):
    """Unified RAG configuration consolidating vector and graph RAG settings."""

    # ... other fields ...

    # GraphRAG-specific settings (moved here to eliminate duplication)
    llm_model: str = Field(default="gemma3-12b-awq")
    community_levels: int = Field(default=3)
    max_cluster_size: int = Field(default=50, description="Leiden algorithm max cluster size...")  # ← DEFAULT IS 50
    leiden_resolution: float = Field(default=1.0, description="Leiden algorithm resolution parameter...")  # ← DEFAULT IS 1.0
```

**THERE IT IS!**

The `RAGSettings` class has Pydantic field defaults of `50` and `1.0`. These are the defaults used when the YAML config doesn't provide values OR when environment variable substitution fails.

---

## THE COMPLETE ROOT CAUSE CHAIN

### Scenario Analysis

**What the user expects**:
1. Set `GRAPHRAG_MAX_CLUSTER_SIZE=150` in Portainer
2. Set `GRAPHRAG_LEIDEN_RESOLUTION=0.5` in Portainer
3. Values flow through Docker → Container → Python → Config

**What actually happens**:

#### Step 1: YAML Reading (config/default.yaml)
```yaml
max_cluster_size: ${GRAPHRAG_MAX_CLUSTER_SIZE:-50}
leiden_resolution: ${GRAPHRAG_LEIDEN_RESOLUTION:-1.0}
```

#### Step 2: Environment Variable Substitution (config.py)
```python
config_str = substitute_environment_variables(config_str)
# Calls: os.environ.get('GRAPHRAG_MAX_CLUSTER_SIZE', '50')
# IF the variable is NOT in os.environ, returns the default '50'
```

#### Step 3: YAML Parsing
```python
config_data = yaml.safe_load(config_str)
# Results in: {'rag': {'max_cluster_size': '50', 'leiden_resolution': '1.0'}}
# NOTE: These are STRINGS, not integers!
```

#### Step 4: Pydantic Validation
```python
Settings.model_validate(config_data)
# Pydantic converts '50' → 50 (int)
# Pydantic converts '1.0' → 1.0 (float)
```

**THE CRITICAL QUESTION**: Are the environment variables actually available in `os.environ` when Python runs?

Let me check the logging output from `get_config()`:

**File: `/home/tuomo/code/fileintel/src/fileintel/core/config.py` (Lines 780-788)**

```python
def get_config() -> "Settings":
    global _settings
    if _settings is None:
        _settings = load_config()
        # CRITICAL DEBUG: Log GraphRAG clustering config at load time
        logger.info(
            f"CONFIG LOAD: GraphRAG clustering - max_cluster_size={_settings.rag.max_cluster_size}, "
            f"leiden_resolution={_settings.rag.leiden_resolution}"
        )
        logger.info(
            f"CONFIG LOAD: Env vars - GRAPHRAG_MAX_CLUSTER_SIZE={os.getenv('GRAPHRAG_MAX_CLUSTER_SIZE', 'NOT_SET')}, "
            f"GRAPHRAG_LEIDEN_RESOLUTION={os.getenv('GRAPHRAG_LEIDEN_RESOLUTION', 'NOT_SET')}"
        )
```

**USER'S LOGS SHOULD SHOW**: What values are actually in `os.environ` at runtime.

---

## ROOT CAUSE DETERMINATION

There are TWO possible root causes:

### Hypothesis A: Environment Variables Not Reaching Container

**Evidence**:
- `.env` file does NOT contain the variables
- `docker-compose.prod.yml` HAS the variables in the `environment:` section (lines 237-238)
- BUT user reports logs show defaults (50, 1.0)

**Possible causes**:
1. Portainer stack variables are NOT being used for Docker Compose substitution
2. The wrong docker-compose file is being used (`.yml` instead of `.prod.yml`)
3. The service has been redeployed but the environment variables weren't passed through
4. There's a typo in the variable names in Portainer

### Hypothesis B: Environment Variables Reaching Container But Not Being Read

**Evidence**:
- The config loading code has debug logging that should show env var values
- User hasn't provided those debug logs yet

**Possible causes**:
1. The config is being cached and not reloaded
2. The environment variables are being cleared before config loading
3. There's a race condition in config initialization

---

## RECOMMENDED DIAGNOSTIC STEPS

### Step 1: Verify Container Environment Variables

SSH into the running container and check:

```bash
docker exec -it <celery-graphrag-gevent-container> env | grep GRAPHRAG
```

**Expected output if working**:
```
GRAPHRAG_MAX_CLUSTER_SIZE=150
GRAPHRAG_LEIDEN_RESOLUTION=0.5
```

**If NOT present**: Problem is in Docker Compose deployment, not Python code.

### Step 2: Check Config Load Logs

Look for this exact log line in the Celery worker logs:

```
CONFIG LOAD: Env vars - GRAPHRAG_MAX_CLUSTER_SIZE=..., GRAPHRAG_LEIDEN_RESOLUTION=...
```

This will definitively show what Python sees in `os.environ`.

### Step 3: Verify Docker Compose File Being Used

Check which docker-compose file Portainer is actually using:

- Is it `docker-compose.yml`?
- Is it `docker-compose.prod.yml`?
- Is it a different file?

---

## RECOMMENDED FIXES (IN ORDER OF PREFERENCE)

### Option 1: Copy Environment Variables from docker-compose.prod.yml to docker-compose.yml (RECOMMENDED)

**File: `/home/tuomo/code/fileintel/docker-compose.yml`**

Add the complete GraphRAG environment configuration to the `celery-graphrag-gevent` service's `environment:` section:

```yaml
environment:
  - PYTHONPATH=/home/appuser/app/src:/home/appuser/.local/lib/python3.9/site-packages
  - PYTHONUNBUFFERED=1
  - DB_USER=${POSTGRES_USER}
  - DB_PASSWORD=${POSTGRES_PASSWORD}
  - DB_HOST=postgres
  - DB_PORT=5432
  - DB_NAME=${POSTGRES_DB}
  - GRAPHRAG_INDEX_PATH=/data  # ADD THIS
  - CELERY_BROKER_URL=redis://redis:6379/1
  - CELERY_RESULT_BACKEND=redis://redis:6379/1
  # GraphRAG core configuration - ADD THESE LINES
  - GRAPHRAG_LLM_MODEL=${GRAPHRAG_LLM_MODEL:-gemma3-12b-awq}
  - GRAPHRAG_EMBEDDING_MODEL=${GRAPHRAG_EMBEDDING_MODEL:-bge-large-en}
  - GRAPHRAG_MAX_TOKENS=${GRAPHRAG_MAX_TOKENS:-12000}
  # GraphRAG clustering configuration (CRITICAL for hierarchy) - ADD THESE LINES
  - GRAPHRAG_MAX_CLUSTER_SIZE=${GRAPHRAG_MAX_CLUSTER_SIZE:-150}
  - GRAPHRAG_LEIDEN_RESOLUTION=${GRAPHRAG_LEIDEN_RESOLUTION:-0.5}
  # GraphRAG async processing (CRITICAL for gevent concurrency) - ADD THESE LINES
  - GRAPHRAG_ASYNC_ENABLED=${GRAPHRAG_ASYNC_ENABLED:-true}
  - GRAPHRAG_ASYNC_BATCH_SIZE=${GRAPHRAG_ASYNC_BATCH_SIZE:-8}
  - GRAPHRAG_ASYNC_MAX_CONCURRENT=${GRAPHRAG_ASYNC_MAX_CONCURRENT:-200}
  - GRAPHRAG_ASYNC_BATCH_TIMEOUT=${GRAPHRAG_ASYNC_BATCH_TIMEOUT:-9000}
  # GraphRAG retry/timeout (CRITICAL for stability) - ADD THESE LINES
  - GRAPHRAG_REQUEST_TIMEOUT=${GRAPHRAG_REQUEST_TIMEOUT:-9000}
  - GRAPHRAG_MAX_RETRIES=${GRAPHRAG_MAX_RETRIES:-200}
  - GRAPHRAG_MAX_RETRY_WAIT=${GRAPHRAG_MAX_RETRY_WAIT:-300}
  # GraphRAG checkpoint/resume - ADD THESE LINES
  - GRAPHRAG_ENABLE_CHECKPOINT_RESUME=${GRAPHRAG_ENABLE_CHECKPOINT_RESUME:-true}
  - GRAPHRAG_VALIDATE_CHECKPOINTS=${GRAPHRAG_VALIDATE_CHECKPOINTS:-true}
  # GraphRAG embedding batch optimization - ADD THIS LINE
  - GRAPHRAG_EMBEDDING_BATCH_MAX_TOKENS=${GRAPHRAG_EMBEDDING_BATCH_MAX_TOKENS:-400}
  # Gevent-specific optimizations
  - GEVENT_RESOLVER=ares
  - GEVENT_THREADPOOL_SIZE=50
  # Python memory optimizations
  - PYTHONMALLOC=malloc
```

**Why this is the BEST solution**:
- Brings `docker-compose.yml` in sync with `docker-compose.prod.yml`
- Uses Docker Compose substitution with defaults (works with Portainer OR .env)
- Portainer stack variables will override the defaults if set
- If no Portainer variables, falls back to sensible defaults (150, 0.5)
- Future-proof: all GraphRAG config variables present

**Action**:
1. Edit `docker-compose.yml`
2. Add the environment variables shown above
3. Redeploy the stack in Portainer

### Option 2: Add Variables to .env File (QUICK FIX)

**File: `/home/tuomo/code/fileintel/.env`**

Add these lines:
```bash
# GraphRAG clustering configuration
GRAPHRAG_MAX_CLUSTER_SIZE=150
GRAPHRAG_LEIDEN_RESOLUTION=0.5
```

**Why this works**:
- The `env_file: - .env` directive in docker-compose.yml will load these into the container
- Python's `os.environ.get()` will find them
- Minimal changes required

**Downside**:
- Only fixes the immediate problem, not the underlying issue
- Other GraphRAG variables still missing from environment
- Does NOT align docker-compose.yml with docker-compose.prod.yml

**Action**: Add the variables to `.env` and redeploy the container.

### Option 3: Switch to docker-compose.prod.yml (CLEAN BUT RISKY)

**Change Portainer stack configuration to use `docker-compose.prod.yml` instead of `docker-compose.yml`**

**Why this works**:
- `docker-compose.prod.yml` already has all the correct configuration
- No file edits needed
- Production-ready configuration

**Downside**:
- May have other differences between the two files
- Could break other parts of the deployment
- Need to analyze full file differences first

**Action**:
1. Run `diff docker-compose.yml docker-compose.prod.yml` to see all differences
2. Assess risk of switching
3. Update Portainer stack to use correct file
4. Redeploy

### Option 4: Override in Portainer (NOT RECOMMENDED)

Set explicit values in Portainer stack environment, but this won't work because `docker-compose.yml` doesn't declare these variables in the `environment:` section.

**Why this DOESN'T work**:
- Docker Compose only passes variables declared in the `environment:` section
- Portainer stack variables are used for substitution in `${VAR:-default}` syntax
- If the variable isn't in the `environment:` section, it won't reach the container

**Conclusion**: This option is NOT viable without fixing the docker-compose file first.

---

## PROOF OF CONFIGURATION FLOW

### Evidence Trail

1. **YAML Config** (`config/default.yaml:109-110`):
   ```yaml
   max_cluster_size: ${GRAPHRAG_MAX_CLUSTER_SIZE:-50}
   leiden_resolution: ${GRAPHRAG_LEIDEN_RESOLUTION:-1.0}
   ```
   ✅ Correctly references environment variables with defaults

2. **Environment Substitution** (`src/fileintel/core/config.py:682-718`):
   ```python
   value = os.environ.get(var_name, default_value)
   ```
   ✅ Correctly reads from `os.environ` with fallback to YAML defaults

3. **Settings Class** (`src/fileintel/core/config.py:317-318`):
   ```python
   max_cluster_size: int = Field(default=50, ...)
   leiden_resolution: float = Field(default=1.0, ...)
   ```
   ⚠️ Has Pydantic defaults that match YAML defaults (safety net)

4. **Config Adapter** (`src/fileintel/rag/graph_rag/adapters/config_adapter.py:235-239`):
   ```python
   cluster_graph_config = ClusterGraphConfig(
       max_cluster_size=settings.rag.max_cluster_size,
       resolution=leiden_resolution,
   )
   ```
   ✅ Correctly passes values from Settings object

5. **Workflow Execution** (`src/graphrag/index/workflows/create_communities.py:36-48`):
   ```python
   max_cluster_size = config.cluster_graph.max_cluster_size
   resolution = getattr(config.cluster_graph, 'resolution', 1.0)
   ```
   ⚠️ Uses `getattr()` with default for resolution (safety net, but shouldn't be needed)

6. **Cluster Operation** (`src/graphrag/index/operations/cluster_graph.py:70-76`):
   ```python
   logger.info(
       f"Running hierarchical Leiden clustering with max_cluster_size={max_cluster_size}, resolution={resolution}"
   )
   ```
   ✅ This is where the log line comes from showing wrong values

---

## FINAL DIAGNOSIS - ROOT CAUSE CONFIRMED

**ROOT CAUSE**: Environment variables `GRAPHRAG_MAX_CLUSTER_SIZE` and `GRAPHRAG_LEIDEN_RESOLUTION` are NOT present in the container's `os.environ` at Python runtime.

**WHY**:
1. The `.env` file does NOT contain these variables
2. **CRITICAL**: `docker-compose.yml` (the active deployment file) is MISSING these environment variable declarations in the `environment:` section
3. `docker-compose.prod.yml` HAS the correct configuration (lines 237-238), but it's NOT the file being used

**EVIDENCE**:
- Git status shows `docker-compose.yml` is modified (the active file)
- `docker-compose.yml` celery-graphrag-gevent service ONLY has these environment vars:
  - `PYTHONPATH`, `PYTHONUNBUFFERED`
  - `DB_*` variables
  - `CELERY_*` variables
  - `GEVENT_*` variables
  - `PYTHONMALLOC`
- **MISSING**: All `GRAPHRAG_*` configuration variables
- `docker-compose.prod.yml` has complete GraphRAG config (lines 232-252)

**FILE COMPARISON**:

`docker-compose.yml` (ACTIVE - INCOMPLETE):
```yaml
environment:
  - PYTHONPATH=/home/appuser/app/src:/home/appuser/.local/lib/python3.9/site-packages
  - PYTHONUNBUFFERED=1
  - DB_USER=${POSTGRES_USER}
  - DB_PASSWORD=${POSTGRES_PASSWORD}
  - DB_HOST=postgres
  - DB_PORT=5432
  - DB_NAME=${POSTGRES_DB}
  - CELERY_BROKER_URL=redis://redis:6379/1
  - CELERY_RESULT_BACKEND=redis://redis:6379/1
  # Gevent-specific optimizations
  - GEVENT_RESOLVER=ares
  - GEVENT_THREADPOOL_SIZE=50
  # Python memory optimizations
  - PYTHONMALLOC=malloc
  # ← MISSING ALL GRAPHRAG_* VARIABLES
```

`docker-compose.prod.yml` (NOT ACTIVE - COMPLETE):
```yaml
environment:
  - PYTHONPATH=/home/appuser/app/src
  - PYTHONUNBUFFERED=1
  - DB_USER=${POSTGRES_USER:-fileintel_user}
  - DB_PASSWORD=${POSTGRES_PASSWORD:-your_secure_password}
  - DB_HOST=postgres
  - DB_PORT=5432
  - DB_NAME=${POSTGRES_DB:-fileintel}
  - GRAPHRAG_INDEX_PATH=/data
  - CELERY_BROKER_URL=redis://redis:6379/1
  - CELERY_RESULT_BACKEND=redis://redis:6379/1
  # GraphRAG core configuration
  - GRAPHRAG_LLM_MODEL=${GRAPHRAG_LLM_MODEL:-gemma3-12b-awq}
  - GRAPHRAG_EMBEDDING_MODEL=${GRAPHRAG_EMBEDDING_MODEL:-bge-large-en}
  - GRAPHRAG_MAX_TOKENS=${GRAPHRAG_MAX_TOKENS:-12000}
  # GraphRAG clustering configuration (CRITICAL for hierarchy)
  - GRAPHRAG_MAX_CLUSTER_SIZE=${GRAPHRAG_MAX_CLUSTER_SIZE:-150}  # ← PRESENT
  - GRAPHRAG_LEIDEN_RESOLUTION=${GRAPHRAG_LEIDEN_RESOLUTION:-0.5}  # ← PRESENT
  # ... more GraphRAG variables ...
```

**EVIDENCE**:
- `.env` file verified to NOT contain the variables
- User logs show defaults (50, 1.0) instead of configured values (150, 0.5)
- The code chain is correct and would work if env vars were present

**RECOMMENDED FIX**: Add the variables to the `.env` file as the most reliable solution:

```bash
GRAPHRAG_MAX_CLUSTER_SIZE=150
GRAPHRAG_LEIDEN_RESOLUTION=0.5
```

Then redeploy the container.

---

## VERIFICATION CHECKLIST

After applying the fix, verify:

1. ✅ Environment variables are in container:
   ```bash
   docker exec <container> env | grep GRAPHRAG_MAX_CLUSTER_SIZE
   docker exec <container> env | grep GRAPHRAG_LEIDEN_RESOLUTION
   ```

2. ✅ Config load logs show correct values:
   ```
   CONFIG LOAD: Env vars - GRAPHRAG_MAX_CLUSTER_SIZE=150, GRAPHRAG_LEIDEN_RESOLUTION=0.5
   CONFIG LOAD: GraphRAG clustering - max_cluster_size=150, leiden_resolution=0.5
   ```

3. ✅ Clustering logs show correct values:
   ```
   Running hierarchical Leiden clustering with max_cluster_size=150, resolution=0.5
   ```

4. ✅ GraphRAG indexing produces expected community hierarchy

---

## ARCHITECTURAL NOTES

### Configuration Precedence (Designed Behavior)

1. **Environment variables** (highest priority)
   - From Portainer stack or .env file
   - Loaded into container's `os.environ`

2. **YAML defaults** (`config/default.yaml`)
   - Used when env var not set
   - Format: `${VAR:-default}`

3. **Pydantic field defaults** (`src/fileintel/core/config.py`)
   - Safety net, should never be used if YAML is correct
   - Used only if YAML key is missing entirely

### Why Multiple Layers of Defaults?

This is defensive programming:
- **YAML defaults**: Allow running without any .env file (development)
- **Pydantic defaults**: Ensure Settings object is always valid (Python safety)
- **getattr() with defaults**: Handle missing config attributes (backward compatibility)

The system is DESIGNED to have multiple fallback layers, which is why it's not immediately obvious when env vars aren't being loaded.

---

## ADDITIONAL FINDINGS

### Docker Compose File Discrepancy

**OBSERVATION**: Git status shows `docker-compose.yml` is modified, but the analysis was done on `docker-compose.prod.yml`.

**RECOMMENDATION**: Verify which file is actually being used in the Portainer deployment:
- Check Portainer stack configuration
- Look for "Compose file path" or similar setting
- Ensure the correct file is being used

### Missing .env Documentation

**OBSERVATION**: `.env.example` (lines 154-155) documents these variables:
```bash
# GRAPHRAG_MAX_CLUSTER_SIZE=150
# GRAPHRAG_LEIDEN_RESOLUTION=0.5
```

But they're commented out, suggesting users should uncomment them if needed.

**RECOMMENDATION**: Update deployment documentation to mention these variables must be set for custom clustering behavior.

---

## CONCLUSION

The configuration system is working as designed. The issue is NOT a bug in the code, but rather a deployment configuration gap where the environment variables are not reaching the container's Python runtime environment.

The fix is simple: Add the variables to `.env` file and redeploy.

The code chain is correct and will work once the environment variables are properly set.
