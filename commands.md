# commands

## Single file

1.  Basic Analysis (Using default_analysis)

This is the most common use case. You send a file, and the system processes it using the prompts located in
prompts/default_analysis.

Command:
curl -X POST -F "file=@C:\path\to\your\document.pdf" http://localhost:8000/api/v1/analyze

Explanation:

- curl -X POST: Specifies that you are making a POST request.
- -F "file=@...": This is the crucial part for file uploads.
- -F tells curl to send the data as multipart/form-data.
- file= matches the parameter name the API is expecting.
- The @ symbol tells curl to use the content of the file at the specified path.
- You must replace C:\path\to\your\document.pdf with the actual, absolute path to the file you want to
  analyze.
- http://localhost:8000/api/v1/analyze: The URL of the endpoint.

2. Custom Analysis (Specifying a Task)

If you have created another prompt directory (e.g., prompts/find_insights), you can tell the API to use it by adding the task_name field.

Command:
curl -X POST -F "file=@C:\path\to\your\document.pdf" -F "task_name=find_insights" http://localhost:8000/api/v1/analyze

Explanation:

- -F "task_name=find_insights": This adds another field to the form data. The API will receive this and pass it to the worker, which will then look for prompts in the prompts/find_insights directory.

What to Expect After You Run the Command
After you submit the request, the API will immediately respond with a JSON object containing the ID of the job it just created. It will look like this:

{
"job_id": "a1b2c3d4-e5f6-7890-g1h2-i3j4k5l6m7n8"
}

This job will be processed in the background by the worker.

How to Check the Result

You can use the job_id from the response to check the status and get the final result.

To check the job status:

curl http://localhost:8000/api/v1/jobs/a1b2c3d4-e5f6-7890-g1h2-i3j4k5l6m7n8/status

To get the final result (once the status is "completed"):

curl http://localhost:8000/api/v1/jobs/a1b2c3d4-e5f6-7890-g1h2-i3j4k5l6m7n8/result

- curl.exe -X POST -F "file=@testfile5.pdf" http://localhost:8000/api/v1/analyze
- curl.exe http://localhost:8000/api/v1/jobs/5bdd1fd9-2c7c-4baf-b39a-f18bc5205fb0/status
- curl.exe http://localhost:8000/api/v1/jobs/5bdd1fd9-2c7c-4baf-b39a-f18bc5205fb0/result
- (Invoke-WebRequest http://localhost:8000/api/v1/jobs/5bdd1fd9-2c7c-4baf-b39a-f18bc5205fb0/result/markdown).Content
- (Invoke-WebRequest http://localhost:8000/api/v1/jobs/5bdd1fd9-2c7c-4baf-b39a-f18bc5205fb0/result/markdown).Content | Out-File result1.md

## Batch

curl.exe -X POST http://localhost:8000/api/v1/batch -H "Content-Type: application/json" -d "{}"

Or you can call it with no body at all.

curl.exe -X POST http://localhost:8000/api/v1/batch

To run a custom analysis (e.g., `find_insights`):
You would send a JSON body specifying the task_name.

curl -X POST http://localhost:8000/api/v1/batch -H "Content-Type: application/json" -d '{"task_name":"find_insights"}'

RAG:
curl.exe -X POST "http://localhost:8000/api/v1/collections?name=ASD" -H "accept: application/json"

curl.exe -X POST "http://localhost:8000/api/v1/collections/6ac841ab-8124-470e-9b74 -cd3c98718ed7/documents" -H "accept: application/json" -F "file=@C:\code\FileIntel\input\input.pdf"

curl.exe -X POST "http://localhost:8000/api/v1/collections/6ac841ab-8124-470e-9b74 │
│ -cd-c3c98718ed7/query" -H "accept: application/json" -H "Content-Type: │
│ application/json" -d "{\\"question\\": \\"\\", \\"task_name\\": │
│ \\"default_analysis\\"}"
