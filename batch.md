# Batch Processing

The batch processing system is designed to analyze multiple files from a designated input directory and save the results to an output directory. This process can be triggered via an API endpoint or a command-line script.

---

### Default Configuration

The default behavior for batch processing is defined in the `config/default.yaml` file under the `batch_processing` section.

```yaml
batch_processing:
  directory_input: "input"
  directory_output: "output"
  default_format: "json"
```

- **`directory_input`**: The folder where the system looks for files to process.
- **`directory_output`**: The folder where the analysis results are saved.
- **`default_format`**: The file format for the output files (e.g., `json`, `markdown`).

When running inside Docker, the `input` and `output` directories in the project root are automatically mapped to the corresponding paths within the container, allowing for easy file management.

---

### How to Run a Batch Job

There are two primary methods for initiating a batch job.

#### 1. Via the API Endpoint (Recommended)

This is the intended method for a running application, as it processes the files asynchronously in the background.

**Step 1: Place Files**
Add one or more files (e.g., PDFs, text files) into the `input` directory.

**Step 2: Trigger the Batch Job**
Make a `POST` request to the `/api/v1/batch` endpoint. You can send an empty JSON body to use the default settings from your configuration file.

**Example using `curl`:**
```bash
curl -X POST http://localhost:8000/api/v1/batch -H "Content-Type: application/json" -d "{}"
```

The API will immediately respond with a `202 Accepted` status and a confirmation message, indicating that the job has started.

```json
{
  "message": "Batch processing job started."
}
```

**Step 3: Retrieve Results**
The system will process each file from the `input` directory. For each processed file, a corresponding result file will be created in the `output` directory (e.g., `my_document_output.json`).

#### 2. Via the Command-Line Script

You can also run the batch process directly from the command line.

**Step 1: Place Files**
As with the API method, place your files in the `input` directory.

**Step 2: Run the Script**
Execute the `batch_process_files.py` script from the project root. If you run it without arguments, it will use the default settings from your configuration file.

```bash
python scripts/batch_process_files.py
```

The script will print its progress to the console and save the results in the `output` directory.

---

### Overriding Default Settings

You can easily override the default configuration for both methods.

#### API Overrides
To override the default settings for an API call, include the desired parameters in the request body.

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/batch \
-H "Content-Type: application/json" \
-d ".*{
      \"input_dir\": \"input/urgent_files\",
      \"output_dir\": \"output/urgent_results\",
      \"output_format\": \"markdown\"
    }"
```

#### CLI Overrides
To override the default settings for the command-line script, use the available options.

**Example:**
```bash
python scripts/batch_process_files.py \
  --input-dir "input/urgent_files" \
  --output-dir "output/urgent_results" \
  --output-format "markdown"
```
