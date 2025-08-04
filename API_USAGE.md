# How to Analyze a Document via the API

This guide provides clear, directly usable commands for analyzing a document.

**Note:** The examples below use the file `testfile.pdf.pdf`. Make sure a file with this exact name exists in the root directory of the project, or replace it with your actual filename in the commands.

---

## For PowerShell Users

### Step 1: Upload the File

Copy and paste this entire block into your PowerShell terminal and press Enter. This command uploads the `testfile.pdf.pdf` and returns a `job_id`.

```powershell
$form = @{ file = Get-Item -Path "testfile.pdf.pdf" }; Invoke-WebRequest -Uri http://localhost:8000/api/v1/analyze -Method POST -Form $form
```

**➡️ Action:** A `job_id` will be returned. **Copy this ID** for the next steps.

### Step 2: Check the Job Status

Replace `<your_job_id>` in the command below with the ID you copied from Step 1. Run the command to see if the job is `pending`, `running`, `completed`, or `failed`.

```powershell
Invoke-WebRequest -Uri http://localhost:8000/api/v1/jobs/9094b901-aa64-4835-8c01-8aa9a6daf89d/status
```

### Step 3: Retrieve the Result

Once the job status is `completed`, run the following command (using the same `job_id`) to get the final analysis.

```powershell
Invoke-WebRequest -Uri http://localhost:8000/api/v1/jobs/<your_job_id>/result
```

---

## For Bash/cURL Users

### Step 1: Upload the File

Run this command to upload `testfile.pdf.pdf`.

```bash
curl -X POST -F "file=@testfile.pdf.pdf" http://localhost:8000/api/v1/analyze
```

**➡️ Action:** A `job_id` will be returned. **Copy this ID** for the next steps.

### Step 2: Check the Job Status

Replace `<your_job_id>` in the command below with the ID you copied from Step 1.

```bash
curl http://localhost:8000/api/v1/jobs/<your_job_id>/status
```

### Step 3: Retrieve the Result

Once the status is `completed`, use the same `job_id` to get the result.

```bash
curl http://localhost:8000/api/v1/jobs/<your_job_id>/result
```

---

## Troubleshooting

If a job fails or you get an error, check the service logs:

```bash
# Check the API service logs
docker-compose logs api

# Check the worker service logs
docker-compose logs worker
```
