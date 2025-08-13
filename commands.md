# commands

## Single file

- curl.exe -X POST -F "file=@testfile5.pdf" http://localhost:8000/api/v1/analyze
- curl.exe http://localhost:8000/api/v1/jobs/5bdd1fd9-2c7c-4baf-b39a-f18bc5205fb0/status
- curl.exe http://localhost:8000/api/v1/jobs/5bdd1fd9-2c7c-4baf-b39a-f18bc5205fb0/result
- (Invoke-WebRequest http://localhost:8000/api/v1/jobs/5bdd1fd9-2c7c-4baf-b39a-f18bc5205fb0/result/markdown).Content
- (Invoke-WebRequest http://localhost:8000/api/v1/jobs/5bdd1fd9-2c7c-4baf-b39a-f18bc5205fb0/result/markdown).Content | Out-File result1.md

## Batch

- curl.exe -X POST http://localhost:8000/api/v1/batch -H "Content-Type: application/json" -d "{}"
- curl.exe http://localhost:8000/api/v1/jobs/fedcba98-7654-3210-fedc-ba9876543210
